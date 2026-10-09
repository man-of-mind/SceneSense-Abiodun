"""Instrumented, semantics-preserving wrapper around the frozen p025 tail.

The production adapter
``rl_agent.splitfusion_live_dispatch_v1.context_tail.ContextualFrozenP025TailAdapter``
records exactly one timing quantity for its whole call
(``camera_pose_reconstruct_ns``) and the qualified edge runtime records exactly
one stage (``frozen_tail``) around it. Both the Phase-13C controlled
``tail_gpu_ms`` and the Phase-15 live ``frozen_tail`` therefore conflate camera
pose reconstruction, ``model.decode_tail``, three finite checks, camera-aware
post-processing, p025 service filtering and the 720x1280 segmentation
upsample/argmax into a single number.

This subclass reproduces the parent's operation sequence *exactly* and adds
only timing. Nothing is reordered, no tensor is copied, no threshold, NMS,
grouping or model call is changed, and the parent's snapshot/serialize handoff
contract is preserved. Two independent guards protect that claim:

* the parent module is pinned by SHA-256, so an edit to the production
  sequence fails this diagnostic closed rather than silently drifting; and
* :func:`assert_parent_equivalence` runs both implementations on the same
  input and requires bit-identical outputs before any measurement is recorded.

CUDA measurement is deliberately non-perturbing. Reusable event pairs are
recorded around every stage and *no* elapsed time is read until
:meth:`resolve` runs after the whole call, so the measured path executes the
same asynchronous CUDA schedule as production. ``decode_tail`` alone is
bracketed by one dedicated pair: the start event is recorded immediately
before ``model.decode_tail(batch, dense=False)`` and the finish event
immediately after it, with nothing else between.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)

from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    ContextTailSnapshot,
    ContextualFrozenP025TailAdapter,
    _require_tree_finite,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import camera_world_matrix
from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError, sha256_file
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import DispatchMetadata


# The exact production tail whose operation sequence this wrapper mirrors.
PARENT_SOURCE_RELPATH = "rl_agent/splitfusion_live_dispatch_v1/context_tail.py"
PARENT_SOURCE_SHA256 = (
    "1781f3013967c464aa8f8de0eb6b46bccb5f3ff00a305c6f6d56012383859fe1"
)

# Wall-clock stage boundaries. The first eight partition the adapter call and
# sum to ``tail_call_total``; ``output_serialization`` is the separate
# serializer stage the edge runtime invokes afterwards.
TAIL_STAGES = (
    "camera_pose_reconstruct",
    "decode_tail_launch",
    "finite_check_outputs",
    "camera_aware_postprocess",
    "finite_check_postprocess",
    "p025_service_filter",
    "finite_check_p025",
    "segmentation_upsample_argmax",
)
SERIALIZE_STAGE = "output_serialization"
TOTAL_STAGE = "tail_call_total"
CUDA_STAGES = (*TAIL_STAGES, "decode_tail_cuda", SERIALIZE_STAGE, TOTAL_STAGE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchContractError(message)


class DeferredCudaStageTimers:
    """Reusable CUDA event pairs whose elapsed times are read only on resolve.

    Recording an event is asynchronous and costs no synchronization, so the
    instrumented path keeps production's CUDA schedule. ``resolve`` performs
    the one explicit completion synchronization and then reads every pair.
    """

    def __init__(self, device: torch.device, stages: tuple[str, ...]) -> None:
        _require(isinstance(device, torch.device), "CUDA timers require a torch.device")
        self._device = device
        self._stages = tuple(stages)
        self._events = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in self._stages
        }
        self._armed: set[str] = set()
        self._values: dict[str, float] = {}

    def reset(self) -> None:
        self._armed.clear()
        self._values.clear()

    def start(self, name: str) -> None:
        _require(name in self._events, f"unknown CUDA stage: {name}")
        _require(name not in self._armed, f"CUDA stage started twice: {name}")
        self._armed.add(name)
        self._events[name][0].record()

    def finish(self, name: str) -> None:
        _require(name in self._armed, f"CUDA stage finished without start: {name}")
        self._events[name][1].record()

    def resolve(self) -> dict[str, float]:
        """Synchronize once, then read every armed pair's device elapsed time."""

        if not self._armed:
            return {}
        torch.cuda.synchronize(self._device)
        for name in sorted(self._armed):
            start, finish = self._events[name]
            value = float(start.elapsed_time(finish))
            _require(
                value == value and value >= 0.0,
                f"invalid CUDA elapsed time for stage {name}: {value}",
            )
            self._values[name] = value
        return dict(self._values)

    @property
    def values(self) -> dict[str, float]:
        return dict(self._values)


class _WallStages:
    """Monotonic-nanosecond stage boundaries recorded in execution order."""

    def __init__(self) -> None:
        self._started: dict[str, int] = {}
        self._elapsed: dict[str, int] = {}
        self.order: list[str] = []

    def reset(self) -> None:
        self._started.clear()
        self._elapsed.clear()
        self.order.clear()

    def start(self, name: str) -> None:
        _require(name not in self._started, f"wall stage started twice: {name}")
        self._started[name] = time.perf_counter_ns()
        self.order.append(name)

    def finish(self, name: str) -> None:
        _require(name in self._started, f"wall stage finished without start: {name}")
        elapsed = time.perf_counter_ns() - self._started[name]
        _require(elapsed >= 0, f"negative wall stage interval: {name}")
        self._elapsed[name] = int(elapsed)

    @property
    def elapsed_ns(self) -> dict[str, int]:
        return dict(self._elapsed)


class InstrumentedFrozenP025TailAdapter(ContextualFrozenP025TailAdapter):
    """The unchanged frozen tail sequence, fully decomposed in time."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        repository_root = Path(__file__).resolve().parents[2]
        observed = sha256_file(repository_root / PARENT_SOURCE_RELPATH)
        _require(
            observed == PARENT_SOURCE_SHA256,
            "frozen tail source drifted from the pinned diagnostic parent: "
            f"{observed} != {PARENT_SOURCE_SHA256}",
        )
        self._cuda = DeferredCudaStageTimers(self._device, CUDA_STAGES)
        self._wall = _WallStages()
        self._resolved: dict[str, Any] | None = None

    # -- measurement plumbing -------------------------------------------------

    def begin_frame(self) -> None:
        self._cuda.reset()
        self._wall.reset()
        self._resolved = None

    def resolve_frame(self) -> dict[str, Any]:
        """One explicit CUDA completion synchronization, then every stage time."""

        cuda_ms = self._cuda.resolve()
        wall_ns = self._wall.elapsed_ns
        record = {
            "wall_ns": wall_ns,
            "cuda_ms": cuda_ms,
            "wall_stage_order": list(self._wall.order),
        }
        self._resolved = record
        return record

    # -- the mirrored production sequence ------------------------------------

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

        self._wall.start(TOTAL_STAGE)
        self._cuda.start(TOTAL_STAGE)

        self._wall.start("camera_pose_reconstruct")
        self._cuda.start("camera_pose_reconstruct")
        camera_pose_reconstruct_started = time.perf_counter_ns()
        camera_world_cpu = camera_world_matrix(context, static["spec"])
        camera_world = torch.tensor(
            camera_world_cpu, dtype=torch.float64, device=self._device
        )
        camera_pose_reconstruct_ns = (
            time.perf_counter_ns() - camera_pose_reconstruct_started
        )
        self._cuda.finish("camera_pose_reconstruct")
        self._wall.finish("camera_pose_reconstruct")

        calibration = {
            "intrinsic": static["intrinsic"],
            "extrinsic": camera_world,
        }
        batch = c2.unsqueeze(0) if c2.ndim == 3 else c2

        # Nothing but the frozen tail call sits between these two events.
        self._wall.start("decode_tail_launch")
        self._cuda.start("decode_tail_cuda")
        outputs = self._model.decode_tail(batch, dense=False)
        self._cuda.finish("decode_tail_cuda")
        self._wall.finish("decode_tail_launch")

        self._wall.start("finite_check_outputs")
        self._cuda.start("finite_check_outputs")
        tensor_count = _require_tree_finite(outputs, "frozen tail output")
        self._cuda.finish("finite_check_outputs")
        self._wall.finish("finite_check_outputs")

        self._wall.start("camera_aware_postprocess")
        self._cuda.start("camera_aware_postprocess")
        postprocessed = self._model.postprocess(outputs, [calibration])[0]
        self._cuda.finish("camera_aware_postprocess")
        self._wall.finish("camera_aware_postprocess")

        self._wall.start("finite_check_postprocess")
        self._cuda.start("finite_check_postprocess")
        tensor_count += _require_tree_finite(
            postprocessed, "camera-aware postprocess output"
        )
        self._cuda.finish("finite_check_postprocess")
        self._wall.finish("finite_check_postprocess")

        self._wall.start("p025_service_filter")
        self._cuda.start("p025_service_filter")
        perception, original_indices = apply_p025_service_policy(
            {"semantic_logits": outputs["semantic_logits"]}, postprocessed
        )
        self._cuda.finish("p025_service_filter")
        self._wall.finish("p025_service_filter")

        self._wall.start("finite_check_p025")
        self._cuda.start("finite_check_p025")
        tensor_count += _require_tree_finite(perception, "p025 service output")
        self._cuda.finish("finite_check_p025")
        self._wall.finish("finite_check_p025")

        source_hw = (static["spec"].source_height, static["spec"].source_width)
        semantic_logits = outputs["semantic_logits"]
        self._wall.start("segmentation_upsample_argmax")
        self._cuda.start("segmentation_upsample_argmax")
        semantic_labels = F.interpolate(
            semantic_logits.float(),
            size=source_hw,
            mode="bilinear",
            align_corners=False,
        ).argmax(1)[0]
        self._cuda.finish("segmentation_upsample_argmax")
        self._wall.finish("segmentation_upsample_argmax")

        self._last = ContextTailSnapshot(
            perception=perception,
            original_indices=original_indices,
            semantic_logits=semantic_logits,
            semantic_labels=semantic_labels,
            outputs=outputs,
            camera_world=camera_world,
            frame_context=context,
            camera_pose_reconstruct_ns=camera_pose_reconstruct_ns,
            output_tensor_count=tensor_count,
        )
        self._cuda.finish(TOTAL_STAGE)
        self._wall.finish(TOTAL_STAGE)
        return perception

    def serialize(self, perception: Mapping[str, torch.Tensor]) -> bytes:
        self._wall.start(SERIALIZE_STAGE)
        self._cuda.start(SERIALIZE_STAGE)
        try:
            return super().serialize(perception)
        finally:
            self._cuda.finish(SERIALIZE_STAGE)
            self._wall.finish(SERIALIZE_STAGE)


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_tree(item) for item in value)
    return value


def _tree_bitwise_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            return False
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return False
        if set(left) != set(right):
            return False
        return all(_tree_bitwise_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not (
            isinstance(left, (tuple, list)) and isinstance(right, (tuple, list))
        ):
            return False
        if len(left) != len(right):
            return False
        return all(_tree_bitwise_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def assert_parent_equivalence(
    *,
    reference: ContextualFrozenP025TailAdapter,
    instrumented: InstrumentedFrozenP025TailAdapter,
    c2: torch.Tensor,
    metadata: DispatchMetadata,
) -> dict[str, Any]:
    """Require bit-identical perception, records and labels from both tails.

    ``reference`` must be an unmodified production adapter over the *same*
    resident model, base and camera registry. Both are driven on one input;
    any numerical, schema or record difference fails closed.
    """

    _require(
        type(reference) is ContextualFrozenP025TailAdapter,
        "equivalence reference must be the unmodified production adapter",
    )
    with torch.inference_mode():
        reference_perception = reference(c2, metadata)
        reference_bytes = reference.serialize(reference_perception)
        reference_snapshot = reference.take_snapshot()
        reference_perception = _clone_tree(reference_perception)
        reference_labels = reference_snapshot.semantic_labels.detach().clone()
        reference_indices = reference_snapshot.original_indices.detach().clone()
        reference_tensor_count = int(reference_snapshot.output_tensor_count)
        del reference_snapshot

        instrumented.begin_frame()
        observed_perception = instrumented(c2, metadata)
        observed_bytes = instrumented.serialize(observed_perception)
        observed_snapshot = instrumented.take_snapshot()
        instrumented.resolve_frame()

    _require(
        _tree_bitwise_equal(reference_perception, observed_perception),
        "instrumented tail perception is not bit-identical to production",
    )
    _require(
        reference_bytes == observed_bytes,
        "instrumented tail service records are not byte-identical to production",
    )
    _require(
        bool(torch.equal(reference_labels, observed_snapshot.semantic_labels)),
        "instrumented tail segmentation labels are not bit-identical to production",
    )
    _require(
        bool(torch.equal(reference_indices, observed_snapshot.original_indices)),
        "instrumented tail p025 indices are not bit-identical to production",
    )
    _require(
        reference_tensor_count == int(observed_snapshot.output_tensor_count),
        "instrumented tail finite-check tensor count drift",
    )
    return {
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "segmentation_labels_bitwise_identical": True,
        "p025_indices_bitwise_identical": True,
        "finite_checked_tensor_count": reference_tensor_count,
        "service_record_bytes": len(reference_bytes),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
    }

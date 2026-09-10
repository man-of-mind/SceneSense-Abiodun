#!/usr/bin/env python3
"""CUDA parity and micro-timing gate for the edge optimization candidate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    guards,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    ContextualFrozenP025TailAdapter,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    STATIC_CAMERA_MODEL_SHA256,
    STATIC_CAMERA_MOUNT_SHA256,
    FrameContextV1,
    Pose6D,
    StaticCameraRegistry,
)
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _Ledger
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for

from .optimized_tail import OptimizedFrozenP025TailAdapter, tree_bitwise_equal


def _elapsed_ms(
    device: torch.device, operation: Any
) -> tuple[float, float, float, Any, bytes, Any]:
    torch.cuda.synchronize(device)
    total_started = time.perf_counter_ns()
    tail_started = total_started
    perception = operation[0](operation[1], operation[2])
    torch.cuda.synchronize(device)
    tail_finished = time.perf_counter_ns()
    serialize_started = tail_finished
    serialized = operation[0].serialize(perception)
    snapshot = operation[0].take_snapshot()
    torch.cuda.synchronize(device)
    finished = time.perf_counter_ns()
    return (
        (finished - total_started) / 1e6,
        (tail_finished - tail_started) / 1e6,
        (finished - serialize_started) / 1e6,
        perception,
        serialized,
        snapshot,
    )


def run(frames: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for optimization qualification")
    if frames < 2:
        raise ValueError("qualification requires at least two frames")
    device = torch.device("cuda:0")
    model, base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    guards.require_frozen_perception([model])
    guards.require_eval_mode([model])
    cameras = StaticCameraRegistry.audited()
    ledger = _Ledger()
    reference = ContextualFrozenP025TailAdapter(
        model=model,
        base=base,
        camera_registry=cameras,
        device=device,
        ledger=ledger,
    )
    candidate = OptimizedFrozenP025TailAdapter(
        model=model,
        base=base,
        camera_registry=cameras,
        device=device,
        ledger=ledger,
    )
    profile = SplitActionRegistry.from_runtime_binding().resolve(71)
    generator = torch.Generator(device=device).manual_seed(20260909)
    reference_ms: list[float] = []
    candidate_ms: list[float] = []
    reference_tail_ms: list[float] = []
    candidate_tail_ms: list[float] = []
    reference_serialize_ms: list[float] = []
    candidate_serialize_ms: list[float] = []
    detection_counts: list[int] = []

    with torch.inference_mode():
        for index in range(frames):
            # A fresh deterministic activation exercises varying score/NMS sets.
            c2 = torch.randn(
                (256, 112, 192),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            context = FrameContextV1(
                stream_id="edge-opt-qualification",
                frame_id=index,
                sequence_id=index + 1,
                capture_timestamp_ns=1_000_000_000 + index * 100_000_000,
                ego_world=Pose6D(
                    -3.9741828441619873 + index * 0.01,
                    28.094629287719727,
                    -0.10292118787765503,
                    -0.048848289996385574,
                    0.1552259624004364,
                    1.5732501745224,
                ),
                camera_model_sha256=STATIC_CAMERA_MODEL_SHA256,
                camera_mount_sha256=STATIC_CAMERA_MOUNT_SHA256,
            )
            metadata = metadata_for(
                profile,
                sequence_id=context.sequence_id,
                capture_timestamp_ns=context.capture_timestamp_ns,
                protocol_version=2,
                frame_context=context,
            )
            # Alternate order to limit first-run/cache bias.
            if index % 2 == 0:
                first = _elapsed_ms(device, (reference, c2, metadata))
                second = _elapsed_ms(device, (candidate, c2, metadata))
                ref, opt = first, second
            else:
                first = _elapsed_ms(device, (candidate, c2, metadata))
                second = _elapsed_ms(device, (reference, c2, metadata))
                opt, ref = first, second
            reference_ms.append(ref[0])
            candidate_ms.append(opt[0])
            reference_tail_ms.append(ref[1])
            candidate_tail_ms.append(opt[1])
            reference_serialize_ms.append(ref[2])
            candidate_serialize_ms.append(opt[2])
            if not tree_bitwise_equal(ref[3], opt[3]):
                mismatches = {}
                for name in sorted(set(ref[3]) | set(opt[3])):
                    left, right = ref[3].get(name), opt[3].get(name)
                    if not (
                        isinstance(left, torch.Tensor)
                        and isinstance(right, torch.Tensor)
                        and left.shape == right.shape
                        and left.dtype == right.dtype
                        and torch.equal(left, right)
                    ):
                        detail: dict[str, Any] = {
                            "reference_shape": None
                            if not isinstance(left, torch.Tensor)
                            else list(left.shape),
                            "candidate_shape": None
                            if not isinstance(right, torch.Tensor)
                            else list(right.shape),
                        }
                        if (
                            isinstance(left, torch.Tensor)
                            and isinstance(right, torch.Tensor)
                            and left.shape == right.shape
                            and left.numel()
                            and left.is_floating_point()
                            and right.is_floating_point()
                        ):
                            detail["maximum_absolute_error"] = float(
                                (left.double() - right.double()).abs().max()
                            )
                        mismatches[name] = detail
                raise RuntimeError(
                    f"perception tensor parity failed at frame {index}: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            if ref[4] != opt[4]:
                raise RuntimeError(f"serialized service parity failed at frame {index}")
            if not torch.equal(ref[5].original_indices, opt[5].original_indices):
                raise RuntimeError(f"p025 index parity failed at frame {index}")
            if not torch.equal(ref[5].semantic_labels, opt[5].semantic_labels):
                raise RuntimeError(f"segmentation parity failed at frame {index}")
            detection_counts.append(int(ref[3]["scores"].numel()))

    warm = slice(1, None)
    reference_median = statistics.median(reference_ms[warm])
    candidate_median = statistics.median(candidate_ms[warm])
    return {
        "schema": "scenesense.splitfusion_edge_optimization_qualification.v1",
        "frames": frames,
        "timed_frames_excluding_first": frames - 1,
        "device": torch.cuda.get_device_name(device),
        "reference_median_ms": reference_median,
        "candidate_median_ms": candidate_median,
        "median_saving_ms": reference_median - candidate_median,
        "speedup": reference_median / candidate_median,
        "reference_tail_median_ms": statistics.median(reference_tail_ms[warm]),
        "candidate_tail_median_ms": statistics.median(candidate_tail_ms[warm]),
        "tail_median_saving_ms": statistics.median(reference_tail_ms[warm])
        - statistics.median(candidate_tail_ms[warm]),
        "reference_serialize_median_ms": statistics.median(
            reference_serialize_ms[warm]
        ),
        "candidate_serialize_median_ms": statistics.median(
            candidate_serialize_ms[warm]
        ),
        "serialize_median_saving_ms": statistics.median(
            reference_serialize_ms[warm]
        )
        - statistics.median(candidate_serialize_ms[warm]),
        "detection_count_range": [min(detection_counts), max(detection_counts)],
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "p025_indices_bitwise_identical": True,
        "segmentation_labels_bitwise_identical": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.frames), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

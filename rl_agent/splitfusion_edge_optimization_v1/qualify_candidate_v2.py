#!/usr/bin/env python3
"""CUDA parity and timing gate for the v2 edge optimization candidate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
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
from .optimized_tail_v2 import OptimizedFrozenP025TailAdapterV2
from .qualify_candidate import _elapsed_ms


def _timed(
    device: torch.device,
    adapter: OptimizedFrozenP025TailAdapter,
    c2: torch.Tensor,
    metadata: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    adapter.begin_frame()
    result = _elapsed_ms(device, (adapter, c2, metadata))
    return result, adapter.resolve_frame()


def _median(values: list[float]) -> float:
    return float(statistics.median(values[1:]))


def run(frames: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v2 optimization qualification")
    if frames < 4:
        raise ValueError("v2 qualification requires at least four frames")
    device = torch.device("cuda:0")
    model, base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    guards.require_frozen_perception([model])
    guards.require_eval_mode([model])
    cameras = StaticCameraRegistry.audited()
    ledger = _Ledger()
    v1 = OptimizedFrozenP025TailAdapter(
        model=model,
        base=base,
        camera_registry=cameras,
        device=device,
        ledger=ledger,
    )
    v2 = OptimizedFrozenP025TailAdapterV2(
        model=model,
        base=base,
        camera_registry=cameras,
        device=device,
        ledger=ledger,
    )
    profile = SplitActionRegistry.from_runtime_binding().resolve(71)
    generator = torch.Generator(device=device).manual_seed(20260910)
    total_ms: dict[str, list[float]] = defaultdict(list)
    tail_ms: dict[str, list[float]] = defaultdict(list)
    serialize_ms: dict[str, list[float]] = defaultdict(list)
    wall_stages: dict[str, dict[str, list[float]]] = {
        "v1": defaultdict(list),
        "v2": defaultdict(list),
    }
    detection_counts: list[int] = []

    with torch.inference_mode():
        for index in range(frames):
            c2 = torch.randn(
                (256, 112, 192),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            context = FrameContextV1(
                stream_id="edge-opt-v2-qualification",
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
            if index % 2 == 0:
                first = _timed(device, v1, c2, metadata)
                second = _timed(device, v2, c2, metadata)
                observed = {"v1": first, "v2": second}
            else:
                first = _timed(device, v2, c2, metadata)
                second = _timed(device, v1, c2, metadata)
                observed = {"v1": second, "v2": first}
            v1_result, v1_timing = observed["v1"]
            v2_result, v2_timing = observed["v2"]
            if not tree_bitwise_equal(v1_result[3], v2_result[3]):
                raise RuntimeError(f"v2 perception parity failed at frame {index}")
            if v1_result[4] != v2_result[4]:
                raise RuntimeError(f"v2 service-byte parity failed at frame {index}")
            if not torch.equal(
                v1_result[5].original_indices, v2_result[5].original_indices
            ):
                raise RuntimeError(f"v2 p025-index parity failed at frame {index}")
            if not torch.equal(
                v1_result[5].semantic_labels, v2_result[5].semantic_labels
            ):
                raise RuntimeError(f"v2 segmentation parity failed at frame {index}")
            for name, result, timing in (
                ("v1", v1_result, v1_timing),
                ("v2", v2_result, v2_timing),
            ):
                total_ms[name].append(float(result[0]))
                tail_ms[name].append(float(result[1]))
                serialize_ms[name].append(float(result[2]))
                for stage, elapsed_ns in timing["wall_ns"].items():
                    wall_stages[name][stage].append(float(elapsed_ns) / 1e6)
            detection_counts.append(int(v1_result[3]["scores"].numel()))

    stage_medians = {
        name: {stage: _median(values) for stage, values in sorted(stages.items())}
        for name, stages in wall_stages.items()
    }
    v1_median = _median(total_ms["v1"])
    v2_median = _median(total_ms["v2"])
    return {
        "schema": "scenesense.splitfusion_edge_optimization_v2_qualification.v1",
        "frames": frames,
        "timed_frames_excluding_first": frames - 1,
        "device": torch.cuda.get_device_name(device),
        "v1_total_median_ms": v1_median,
        "v2_total_median_ms": v2_median,
        "total_median_saving_ms": v1_median - v2_median,
        "total_speedup": v1_median / v2_median,
        "v1_tail_median_ms": _median(tail_ms["v1"]),
        "v2_tail_median_ms": _median(tail_ms["v2"]),
        "tail_median_saving_ms": _median(tail_ms["v1"])
        - _median(tail_ms["v2"]),
        "v1_serialize_median_ms": _median(serialize_ms["v1"]),
        "v2_serialize_median_ms": _median(serialize_ms["v2"]),
        "stage_medians_ms": stage_medians,
        "detection_count_range": [min(detection_counts), max(detection_counts)],
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "p025_indices_bitwise_identical": True,
        "segmentation_labels_bitwise_identical": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args.frames), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

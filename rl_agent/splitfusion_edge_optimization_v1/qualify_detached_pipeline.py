#!/usr/bin/env python3
"""CUDA parity and overlap qualification for the detached tail handoff."""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import threading
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
from rl_agent.splitfusion_edge_freshness_scheduler_v1.pipeline import (
    BoundedTwoStagePipeline,
    CandidatePolicy,
    PipelineConfig,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.scheduler import (
    FrameTicket,
    TerminalReason,
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
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import DispatchMetadata, metadata_for

from .detached_tail import DetachedOptimizedTailAdapter, DetachedTailWorkProduct
from .optimized_tail import OptimizedFrozenP025TailAdapter, tree_bitwise_equal


def _context(index: int, *, stream_id: str, capture_ns: int) -> FrameContextV1:
    return FrameContextV1(
        stream_id=stream_id,
        frame_id=index,
        sequence_id=index + 1,
        capture_timestamp_ns=capture_ns,
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


@functools.lru_cache(maxsize=1)
def _profile() -> Any:
    return SplitActionRegistry.from_runtime_binding().resolve(71)


def _metadata(index: int, *, stream_id: str, capture_ns: int) -> DispatchMetadata:
    context = _context(index, stream_id=stream_id, capture_ns=capture_ns)
    return metadata_for(
        _profile(),
        sequence_id=context.sequence_id,
        capture_timestamp_ns=context.capture_timestamp_ns,
        protocol_version=2,
        frame_context=context,
    )


def _make_tail(
    kind: type[OptimizedFrozenP025TailAdapter],
    *,
    model: torch.nn.Module,
    base: Any,
    cameras: StaticCameraRegistry,
    device: torch.device,
) -> OptimizedFrozenP025TailAdapter:
    return kind(
        model=model,
        base=base,
        camera_registry=cameras,
        device=device,
        ledger=_Ledger(),
    )


def _parity(
    *,
    frames: int,
    device: torch.device,
    model: torch.nn.Module,
    base: Any,
    cameras: StaticCameraRegistry,
) -> dict[str, Any]:
    reference = _make_tail(
        OptimizedFrozenP025TailAdapter,
        model=model,
        base=base,
        cameras=cameras,
        device=device,
    )
    detached = _make_tail(
        DetachedOptimizedTailAdapter,
        model=model,
        base=base,
        cameras=cameras,
        device=device,
    )
    assert isinstance(detached, DetachedOptimizedTailAdapter)
    generator = torch.Generator(device=device).manual_seed(20260910)
    reference_ms: list[float] = []
    detached_compute_ms: list[float] = []
    detached_publication_ms: list[float] = []
    record_counts: list[int] = []
    with torch.inference_mode():
        for index in range(frames):
            c2 = torch.randn(
                (256, 112, 192),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            metadata = _metadata(
                index,
                stream_id="detached-parity",
                capture_ns=1_000_000_000 + index * 100_000_000,
            )

            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            reference_perception = reference(c2, metadata)
            reference_serialized = reference.serialize(reference_perception)
            reference_snapshot = reference.take_snapshot()
            torch.cuda.synchronize(device)
            reference_ms.append((time.perf_counter_ns() - started) / 1e6)

            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            product = detached.compute_product(c2, metadata)
            torch.cuda.synchronize(device)
            compute_finished = time.perf_counter_ns()
            serialized = detached.serialize_product(product)
            publication_finished = time.perf_counter_ns()
            detached_compute_ms.append((compute_finished - started) / 1e6)
            detached_publication_ms.append(
                (publication_finished - compute_finished) / 1e6
            )

            if not tree_bitwise_equal(reference_perception, product.perception):
                raise RuntimeError(f"detached perception parity failed at frame {index}")
            if reference_serialized != serialized.serialized_records:
                raise RuntimeError(f"detached serialization parity failed at frame {index}")
            if not torch.equal(
                reference_snapshot.original_indices, product.original_indices
            ):
                raise RuntimeError(f"detached p025 index parity failed at frame {index}")
            if not torch.equal(
                reference_snapshot.semantic_labels, product.semantic_labels
            ):
                raise RuntimeError(f"detached segmentation parity failed at frame {index}")
            record_counts.append(serialized.record_count)

    warm = slice(1, None)
    return {
        "frames": frames,
        "timed_frames_excluding_first": frames - 1,
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "p025_indices_bitwise_identical": True,
        "segmentation_labels_bitwise_identical": True,
        "reference_service_median_ms": statistics.median(reference_ms[warm]),
        "detached_compute_median_ms": statistics.median(detached_compute_ms[warm]),
        "detached_cpu_publication_median_ms": statistics.median(
            detached_publication_ms[warm]
        ),
        "record_count_range": [min(record_counts), max(record_counts)],
    }


def _pipeline_policy(
    *,
    policy: CandidatePolicy,
    frames: int,
    interval_ms: float,
    device: torch.device,
    model: torch.nn.Module,
    base: Any,
    cameras: StaticCameraRegistry,
) -> dict[str, Any]:
    detached = _make_tail(
        DetachedOptimizedTailAdapter,
        model=model,
        base=base,
        cameras=cameras,
        device=device,
    )
    assert isinstance(detached, DetachedOptimizedTailAdapter)
    generator = torch.Generator(device=device).manual_seed(20260911)
    inputs = [
        torch.randn(
            (256, 112, 192),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for _ in range(frames)
    ]
    torch.cuda.synchronize(device)
    compute_threads: set[int] = set()
    publication_threads: set[int] = set()
    published: list[int] = []

    def compute(
        _ticket: FrameTicket,
        payload: tuple[torch.Tensor, DispatchMetadata],
    ) -> DetachedTailWorkProduct:
        compute_threads.add(threading.get_ident())
        return detached.compute_product(payload[0], payload[1])

    def publish(ticket: FrameTicket, product: DetachedTailWorkProduct) -> None:
        publication_threads.add(threading.get_ident())
        detached.serialize_product(product)
        published.append(ticket.sequence_id)

    pipeline_config = (
        PipelineConfig(
            policy=policy,
            processing_horizon_ns=500_000_000,
            initial_predicted_compute_ns=30_000_000,
            initial_predicted_publication_ns=3_000_000,
            predicted_post_publication_install_ns=15_000_000,
        )
        if policy is CandidatePolicy.PREDICTED_INSTALL_HORIZON
        else PipelineConfig(
            policy=policy,
            processing_horizon_ns=500_000_000,
        )
    )
    pipeline = BoundedTwoStagePipeline(
        config=pipeline_config,
        compute=compute,
        publish=publish,
    )
    pipeline.start()
    monotonic_started = time.monotonic_ns()
    interval_ns = int(interval_ms * 1e6)
    for index, c2 in enumerate(inputs):
        target = monotonic_started + index * interval_ns
        while time.monotonic_ns() < target:
            time.sleep(min(0.001, max(0.0, (target - time.monotonic_ns()) / 1e9)))
        capture_ns = time.time_ns()
        metadata = _metadata(
            index,
            stream_id=f"pipeline-{policy.value.lower()}",
            capture_ns=capture_ns,
        )
        frame_context = metadata.frame_context
        assert frame_context is not None
        ticket = FrameTicket(
            run_id="detached-pipeline-qualification",
            cell_id=policy.value.lower(),
            stream_id=frame_context.stream_id,
            frame_id=frame_context.frame_id,
            sequence_id=frame_context.sequence_id,
            action_id=metadata.action_id,
            capture_timestamp_ns=capture_ns,
            edge_arrival_timestamp_ns=time.time_ns(),
            feature_bytes=6464,
        )
        pipeline.offer(ticket, (c2, metadata))
    outcomes = pipeline.close_and_join(timeout_s=60.0)
    torch.cuda.synchronize(device)
    snapshot = pipeline.snapshot()
    reasons = {
        reason.value: sum(item.reason is reason for item in outcomes)
        for reason in TerminalReason
    }
    if len(compute_threads) != 1 or len(publication_threads) != 1:
        raise RuntimeError("pipeline did not preserve single stage owners")
    if compute_threads == publication_threads:
        raise RuntimeError("compute and publication used the same owner")
    if len(outcomes) != frames:
        raise RuntimeError("pipeline terminal accounting did not reconcile")
    return {
        "policy": policy.value,
        "frames_offered": frames,
        "frames_published": len(published),
        "terminal_reason_counts": reasons,
        "compute_pending_high_water": snapshot.compute_pending_high_water,
        "publication_pending_high_water": snapshot.publication_pending_high_water,
        "maximum_active_stage_workers": snapshot.maximum_active_stage_workers,
        "stage_overlap_observed": snapshot.stage_overlap_observed,
        "single_compute_owner": len(compute_threads) == 1,
        "single_publication_owner": len(publication_threads) == 1,
        "distinct_stage_owners": compute_threads != publication_threads,
    }


def run(*, parity_frames: int, pipeline_frames: int, interval_ms: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for detached-pipeline qualification")
    if parity_frames < 2:
        raise ValueError("parity qualification requires at least two frames")
    if pipeline_frames < 4:
        raise ValueError("pipeline qualification requires at least four frames")
    if interval_ms <= 0.0:
        raise ValueError("pipeline interval must be positive")
    device = torch.device("cuda:0")
    model, base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    guards.require_frozen_perception([model])
    guards.require_eval_mode([model])
    _per_tensor, state_before = phase11b.common.state_hashes(model)
    cameras = StaticCameraRegistry.audited()
    parity = _parity(
        frames=parity_frames,
        device=device,
        model=model,
        base=base,
        cameras=cameras,
    )
    policies = [
        _pipeline_policy(
            policy=policy,
            frames=pipeline_frames,
            interval_ms=interval_ms,
            device=device,
            model=model,
            base=base,
            cameras=cameras,
        )
        for policy in CandidatePolicy
    ]
    _per_tensor_after, state_after = phase11b.common.state_hashes(model)
    if state_before != state_after:
        raise RuntimeError("frozen perception state changed during qualification")
    overlap_observed = any(
        bool(item["stage_overlap_observed"]) for item in policies
    )
    return {
        "schema": "scenesense.splitfusion.detached_edge_pipeline_qualification.v1",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parity": parity,
        "pipeline_interval_ms": interval_ms,
        "policies": policies,
        "frozen_state_sha256_before": state_before,
        "frozen_state_sha256_after": state_after,
        "frozen_state_unchanged": True,
        "qualification_status": (
            "PARITY_ACCOUNTING_PASS_OVERLAP_OBSERVED"
            if overlap_observed
            else "PARITY_ACCOUNTING_PASS_OVERLAP_NOT_OBSERVED"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-frames", type=int, default=6)
    parser.add_argument("--pipeline-frames", type=int, default=12)
    parser.add_argument("--pipeline-interval-ms", type=float, default=5.0)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                parity_frames=args.parity_frames,
                pipeline_frames=args.pipeline_frames,
                interval_ms=args.pipeline_interval_ms,
            ),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

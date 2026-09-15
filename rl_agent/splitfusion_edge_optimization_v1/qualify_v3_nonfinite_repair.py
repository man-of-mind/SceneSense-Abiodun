#!/usr/bin/env python3
"""Qualification gate for the v3 deferred-finite-validation repair.

The action-15 (noAE/UINT4/q0.70) live cells stopped fail-closed on
``v3 camera-aware postprocess output contains non-finite values`` while the
identical v2 geometry path never did.  The difference between the two is not
the arithmetic — it is that v3 evaluates the finite predicate on a second CUDA
stream and reads the verdict back later.

This gate establishes three things on *real* reconstructed C2, not on synthetic
noise:

1. ``reachability`` — the frozen postprocess tree stays finite with tens of
   orders of magnitude of headroom, so a genuine non-finite output is not
   reachable from a finite ``decode_tail`` tree;
2. ``allocator_hazard`` — the deferred validator no longer returns a verdict
   derived from storage the caching allocator recycled underneath it; and
3. ``equivalence`` — repaired v3 is bit-identical to both the frozen reference
   tail and the qualified v2 adapter on every registered output surface.

Nothing here clamps, masks or rewrites an output.  Injected non-finite values
must still stop the frame.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    collate_batch,
    encode_front,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    STATIC_CAMERA_MODEL_SHA256,
    STATIC_CAMERA_MOUNT_SHA256,
    FrameContextV1,
    Pose6D,
    camera_world_matrix,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.timing import (
    EDGE_STAGES,
    UE_STAGES,
    StageRecorder,
)
from rl_agent.splitfusion_live_dispatch_v1.transport import ProductionSplitCodec
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for

from .detached_edge_preload_v3 import preload_detached_optimized_edge_v3
from .optimized_tail import tree_bitwise_equal
from .optimized_tail_v2 import (
    OptimizedFrozenP025TailAdapterV2,
    postprocess_geometry_after_nms_v2,
)
from .optimized_tail_v3 import (
    OptimizedFrozenP025TailAdapterV3,
    _DeferredFiniteValidator,
)

# Every floating-point field the camera-aware geometry stage emits.
GEOMETRY_FIELDS = (
    "boxes",
    "scores",
    "local_xyz",
    "world_xyz",
    "dimensions",
    "log_dimensions",
    "yaw",
    "raw_yaw",
    "yaw_raw_norm",
    "physical_uv",
    "physical_ray_offsets",
    "depth",
    "depth_residual",
    "depth_bin_logits",
    "depth_bin_probabilities",
    "bounded_depth_residuals",
)

# ``dimensions = exp(log_dimensions)`` in float64 overflows above this; the
# yaw norm is float32. Both are reported against the measured maxima.
LOG_DIMENSION_OVERFLOW = 709.78
FLOAT32_OVERFLOW = 3.4028235e38


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _context(index: int) -> FrameContextV1:
    return FrameContextV1(
        stream_id="v3-nonfinite-repair-qualification",
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


def allocator_hazard_trials(
    device: torch.device, trials: int, elements: int = 1 << 22
) -> dict[str, Any]:
    """Drop a checked tensor before the validation stream has read it.

    Before the repair the allocator was free to hand that block to the very
    next compute-stream allocation, so the deferred verdict described whatever
    replaced it. Every trial here checks an all-zero tensor, so a non-finite
    verdict can only come from recycled storage.
    """

    validator = _DeferredFiniteValidator(device)
    false_verdicts = 0
    for _ in range(trials):
        checked = torch.zeros(elements, device=device, dtype=torch.float32)
        wide = torch.zeros((64, elements // 64), device=device, dtype=torch.float32)
        validator.launch({"checked": checked, "wide": wide}, "allocator hazard probe")
        del checked, wide
        poison = [
            torch.full((elements,), float("nan"), device=device, dtype=torch.float32)
            for _ in range(2)
        ]
        try:
            validator.resolve()
        except RuntimeError:
            false_verdicts += 1
            validator.discard()
        del poison
    return {
        "trials": trials,
        "false_nonfinite_verdicts": false_verdicts,
        "asynchronous_verdict_corrections": (
            validator.asynchronous_verdict_corrections
        ),
        "passed": false_verdicts == 0,
    }


def injected_nonfinite_still_rejected(device: torch.device) -> dict[str, Any]:
    validator = _DeferredFiniteValidator(device)
    validator.launch(
        {"injected": torch.tensor([float("nan")], device=device)},
        "injected qualification fault",
    )
    rejected = False
    try:
        validator.resolve()
    except RuntimeError as exc:
        rejected = "non-finite" in str(exc)
    _require(rejected, "repaired validator accepted an injected NaN")
    return {"rejected": True, "passed": True}


def run(frames: int, actions: tuple[int, ...], hazard_trials: int) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for this qualification")
    device = torch.device("cuda:0")
    edge = preload_detached_optimized_edge_v3(device)
    model, base = edge.model, edge.base
    ranker = phase11b._load_ranker(device)
    dataset = build_train_dataset(base)
    codec = ProductionSplitCodec()
    registry = SplitActionRegistry.from_runtime_binding()

    reference = edge.reference_tail
    v2 = OptimizedFrozenP025TailAdapterV2(
        model=model, base=base, camera_registry=edge.camera_registry,
        device=device, ledger=edge.ledger,
    )
    v3 = OptimizedFrozenP025TailAdapterV3(
        model=model, base=base, camera_registry=edge.camera_registry,
        device=device, ledger=edge.ledger,
    )
    static = v3._static[(STATIC_CAMERA_MODEL_SHA256, STATIC_CAMERA_MOUNT_SHA256)]

    generator = torch.Generator().manual_seed(20260914)
    rows = torch.randperm(len(dataset.rows), generator=generator)[:frames].tolist()

    per_action: dict[str, Any] = {}
    with torch.inference_mode():
        for action in actions:
            profile = registry.resolve(action)
            autoencoder = edge.autoencoders.get(profile.family)
            decoder = None if profile.family == "noAE" else autoencoder
            headroom = {field: 0.0 for field in GEOMETRY_FIELDS}
            nonfinite_events: list[dict[str, Any]] = []
            compared = 0
            for index, row in enumerate(rows):
                batch = collate_batch(base, dataset, [row])
                c2 = encode_front(model, batch, device)[0]
                payload = codec.encode(
                    profile, c2,
                    ranker=(None if profile.q_e4 == 0 else ranker),
                    ae_encoder=autoencoder,
                    timing=StageRecorder(UE_STAGES),
                )
                inspected = codec.inspect(payload, timing=StageRecorder(EDGE_STAGES))
                recon = codec.decode(
                    inspected, decoder=decoder, tail_device=device,
                    timing=StageRecorder(EDGE_STAGES),
                ).c2
                context = _context(index)
                metadata = metadata_for(
                    profile,
                    sequence_id=context.sequence_id,
                    capture_timestamp_ns=context.capture_timestamp_ns,
                    protocol_version=2,
                    frame_context=context,
                )

                # Reachability: the geometry tree, measured directly.
                extrinsic = torch.tensor(
                    camera_world_matrix(context, static["spec"]),
                    dtype=torch.float64, device=device,
                )
                outputs = model.decode_tail(recon.unsqueeze(0), dense=False)
                postprocessed = postprocess_geometry_after_nms_v2(
                    model, outputs,
                    [{"intrinsic": static["intrinsic"], "extrinsic": extrinsic}],
                )[0]
                for field in GEOMETRY_FIELDS:
                    tensor = postprocessed[field]
                    if tensor.numel() == 0:
                        continue
                    finite = tensor[torch.isfinite(tensor)]
                    if finite.numel() != tensor.numel():
                        nonfinite_events.append(
                            {"frame": index, "row": int(row), "field": field}
                        )
                    if finite.numel():
                        headroom[field] = max(
                            headroom[field], float(finite.abs().max())
                        )
                del outputs, postprocessed

                # Equivalence on every registered output surface.
                reference_perception = reference(recon, metadata)
                reference_bytes = reference.serialize(reference_perception)
                reference_snapshot = reference.take_snapshot()
                v2_perception = v2(recon, metadata)
                v2_bytes = v2.serialize(v2_perception)
                v2_snapshot = v2.take_snapshot()
                v3_perception = v3(recon, metadata)
                v3_bytes = v3.serialize(v3_perception)
                v3_snapshot = v3.take_snapshot()

                _require(
                    tree_bitwise_equal(reference_perception, v3_perception),
                    f"action {action} frame {index}: v3/reference perception drift",
                )
                _require(
                    tree_bitwise_equal(v2_perception, v3_perception),
                    f"action {action} frame {index}: v3/v2 perception drift",
                )
                _require(
                    reference_bytes == v3_bytes == v2_bytes,
                    f"action {action} frame {index}: service-record byte drift",
                )
                _require(
                    torch.equal(
                        reference_snapshot.original_indices,
                        v3_snapshot.original_indices,
                    )
                    and torch.equal(
                        v2_snapshot.original_indices, v3_snapshot.original_indices
                    ),
                    f"action {action} frame {index}: p025 index drift",
                )
                _require(
                    torch.equal(
                        reference_snapshot.semantic_labels,
                        v3_snapshot.semantic_labels,
                    )
                    and torch.equal(
                        v2_snapshot.semantic_labels, v3_snapshot.semantic_labels
                    ),
                    f"action {action} frame {index}: segmentation label drift",
                )
                compared += 1

            per_action[str(action)] = {
                "profile_id": profile.profile_id,
                "family": profile.family,
                "quantizer": profile.quantizer,
                "q": profile.q,
                "frames_compared": compared,
                "perception_bitwise_identical": True,
                "service_records_byte_identical": True,
                "p025_indices_bitwise_identical": True,
                "segmentation_labels_bitwise_identical": True,
                "nonfinite_events": nonfinite_events,
                "geometry_finite_absmax": {
                    name: round(value, 6) for name, value in headroom.items()
                },
                "log_dimensions_overflow_threshold": LOG_DIMENSION_OVERFLOW,
                "log_dimensions_headroom_decades": round(
                    LOG_DIMENSION_OVERFLOW / max(headroom["log_dimensions"], 1e-12), 3
                ),
                "float32_overflow_threshold": FLOAT32_OVERFLOW,
                "yaw_norm_headroom_decades": round(
                    FLOAT32_OVERFLOW / max(headroom["yaw_raw_norm"], 1e-12), 3
                ),
            }

    result = {
        "schema": "scenesense.splitfusion_v3_nonfinite_repair_qualification.v1",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "frames_per_action": frames,
        "actions": list(actions),
        "reachability": per_action,
        "allocator_hazard": allocator_hazard_trials(device, hazard_trials),
        "injected_nonfinite": injected_nonfinite_still_rejected(device),
        "v3_asynchronous_verdict_corrections": v3.asynchronous_verdict_corrections,
    }
    result["passed"] = (
        result["allocator_hazard"]["passed"]
        and result["injected_nonfinite"]["passed"]
        and result["v3_asynchronous_verdict_corrections"] == 0
        and all(
            not entry["nonfinite_events"] and entry["frames_compared"] == frames
            for entry in per_action.values()
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--actions", type=str, default="15,30,50,71")
    parser.add_argument("--hazard-trials", type=int, default=400)
    args = parser.parse_args()
    report = run(
        args.frames,
        tuple(int(value) for value in args.actions.split(",")),
        args.hazard_trials,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

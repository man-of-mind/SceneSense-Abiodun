#!/usr/bin/env python3
"""Full resident UE/edge parity and timing gate for the v3 candidate."""

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
from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import (
    preload_detached_optimized_edge_v3,
)
from rl_agent.splitfusion_edge_optimization_v1.optimized_tail import (
    tree_bitwise_equal,
)
from rl_agent.splitfusion_live_dispatch_v1.envelope import unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES, StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    ProductionSplitCodec,
    require_inner_agreement,
)
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for
from rl_agent.splitfusion_timing_diagnostic_v1.edge_preload import preload_ue


ACTIONS = (30, 15, 50, 71)


def _decode(
    edge: Any,
    wire: bytes,
    action_id: int,
    codec: ProductionSplitCodec,
) -> tuple[torch.Tensor, Any, dict[str, int]]:
    outer = unpack_envelope(wire)
    profile = edge.registry.resolve(action_id)
    if int(outer.action_id) != action_id or outer.frame_context is None:
        raise RuntimeError("resident v3 qualification envelope identity drift")
    timing = StageRecorder(EDGE_STAGES)
    inspected = codec.inspect(outer.inner_payload, timing=timing)
    require_inner_agreement(profile, inspected.identity)
    decoder = (
        None
        if profile.family == "noAE"
        else edge.runtime._ae_decoders.get(profile.family)
    )
    with torch.inference_mode():
        decoded = codec.decode(
            inspected,
            decoder=decoder,
            tail_device=edge.device,
            timing=timing,
        )
    metadata = metadata_for(
        profile,
        sequence_id=outer.sequence_id,
        capture_timestamp_ns=outer.capture_timestamp_ns,
        protocol_version=outer.protocol_version,
        frame_context=outer.frame_context,
    )
    stages = {
        boundary.name: (
            boundary.finished_monotonic_ns - boundary.started_monotonic_ns
        )
        for boundary in timing.snapshot().boundaries
    }
    return decoded.c2, metadata, stages


def _state_hashes(models: list[torch.nn.Module]) -> list[str]:
    return [phase11b.common.state_hashes(model)[1] for model in models]


def run(*, frames_per_action: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if frames_per_action < 2:
        raise ValueError("at least two frames per action are required")
    device = torch.device("cuda:0")
    ue, _ledger, ue_models, _base, _registry = preload_ue(device)
    edge = preload_detached_optimized_edge_v3(device)
    all_models = [edge.model, *edge.autoencoders.values(), *ue_models]
    state_before = _state_hashes(all_models)
    generator = torch.Generator(device=device).manual_seed(20260910)
    rows: list[dict[str, Any]] = []
    production_codec = ProductionSplitCodec()
    candidate_codec = edge.runtime._codec

    for action_id in ACTIONS:
        profile = edge.registry.resolve(action_id)
        stream_id = f"live-v3-cuda-a{action_id}"
        for index in range(frames_per_action):
            sequence = index + 1
            capture_ns = time.time_ns()
            input_tensor = torch.randn(
                (1, 7, 448, 768),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                prepared = ue.prepare(
                    action_id,
                    input_tensor,
                    sequence_id=sequence,
                    capture_timestamp_ns=capture_ns,
                    frame_context=build_frame_context_v1(
                        stream_id=stream_id,
                        frame_id=sequence,
                        sequence_id=sequence,
                        capture_timestamp_ns=capture_ns,
                        ego_world_x=float(index) * 0.1,
                        ego_world_y=0.0,
                        ego_world_z=0.0,
                        ego_world_pitch=0.0,
                        ego_world_yaw=0.0,
                        ego_world_roll=0.0,
                    ),
                )
            wire = bytes(prepared.wire_bytes)
            torch.cuda.synchronize(device)
            production_started = time.perf_counter_ns()
            production_c2, metadata, production_stages = _decode(
                edge, wire, action_id, production_codec
            )
            torch.cuda.synchronize(device)
            production_decode_ms = (time.perf_counter_ns() - production_started) / 1e6
            candidate_started = time.perf_counter_ns()
            candidate_c2, candidate_metadata, candidate_stages = _decode(
                edge, wire, action_id, candidate_codec
            )
            torch.cuda.synchronize(device)
            candidate_decode_ms = (time.perf_counter_ns() - candidate_started) / 1e6
            if not torch.equal(production_c2, candidate_c2):
                raise RuntimeError("v3 reconstructed-C2 parity failed")
            if metadata != candidate_metadata:
                raise RuntimeError("v3 decode metadata parity failed")

            with torch.inference_mode():
                reference = edge.reference_tail(production_c2, metadata)
                reference_bytes = edge.reference_tail.serialize(reference)
                reference_snapshot = edge.reference_tail.take_snapshot()
                torch.cuda.synchronize(device)
                service_started = time.perf_counter_ns()
                edge.tail.begin_frame()
                computed = edge.runtime.process_compute(
                    wire, transmitted_action_id=action_id
                )
                stage_timings = edge.tail.resolve_frame()
                published = edge.runtime.publish_cpu(computed)
                torch.cuda.synchronize(device)
                candidate_service_ms = (time.perf_counter_ns() - service_started) / 1e6
            if not tree_bitwise_equal(reference, computed.work.perception):
                raise RuntimeError("v3 resident perception parity failed")
            if reference_bytes != published.serialized.serialized_records:
                raise RuntimeError("v3 resident serialization parity failed")
            if not torch.equal(
                reference_snapshot.original_indices, computed.work.original_indices
            ):
                raise RuntimeError("v3 resident p025-index parity failed")
            if not torch.equal(
                reference_snapshot.semantic_labels, computed.work.semantic_labels
            ):
                raise RuntimeError("v3 resident segmentation parity failed")
            rows.append(
                {
                    "action_id": action_id,
                    "profile_id": profile.profile_id,
                    "frame_id": sequence,
                    "wire_bytes": len(wire),
                    "reconstructed_c2_bitwise_identical": True,
                    "production_decode_ms": production_decode_ms,
                    "candidate_decode_ms": candidate_decode_ms,
                    "production_ae_decode_stage_ms": production_stages["ae_decode"] / 1e6,
                    "candidate_ae_decode_stage_ms": candidate_stages["ae_decode"] / 1e6,
                    "candidate_service_ms": candidate_service_ms,
                    "candidate_tail_total_ms": stage_timings["wall_ns"][
                        "tail_call_total"
                    ]
                    / 1e6,
                    "service_record_bytes": len(
                        published.serialized.serialized_records
                    ),
                    "service_record_count": published.serialized.record_count,
                }
            )

    state_after = _state_hashes(all_models)
    if state_before != state_after:
        raise RuntimeError("frozen model state changed")
    summaries: dict[str, Any] = {}
    for action_id in ACTIONS:
        subset = [row for row in rows if row["action_id"] == action_id][1:]
        summaries[str(action_id)] = {
            name: statistics.median(float(row[name]) for row in subset)
            for name in (
                "production_decode_ms",
                "candidate_decode_ms",
                "candidate_service_ms",
                "candidate_tail_total_ms",
            )
        }
        summaries[str(action_id)]["decode_saving_ms"] = (
            summaries[str(action_id)]["production_decode_ms"]
            - summaries[str(action_id)]["candidate_decode_ms"]
        )
    return {
        "schema": "scenesense.splitfusion_edge_optimization_v3_resident_gate.v1",
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "actions": list(ACTIONS),
        "frames_per_action": frames_per_action,
        "summaries_ms": summaries,
        "rows": rows,
        "reconstructed_c2_bitwise_identical": True,
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "p025_indices_bitwise_identical": True,
        "segmentation_labels_bitwise_identical": True,
        "all_frozen_states_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-action", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run(frames_per_action=args.frames_per_action), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CUDA-only SFD1 parity gate for the live freshness edge candidate."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload import (
    preload_detached_optimized_edge,
)
from rl_agent.splitfusion_edge_optimization_v1.optimized_tail import (
    tree_bitwise_equal,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_timing_diagnostic_v1.edge_preload import preload_ue
from rl_agent.splitfusion_timing_diagnostic_v1.edge_service import (
    _decode_for_equivalence,
)


def run(*, frames_per_action: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if frames_per_action < 2:
        raise ValueError("at least two frames per action are required")
    device = torch.device("cuda:0")
    ue, _ledger, ue_models, _base, _registry = preload_ue(device)
    edge = preload_detached_optimized_edge(device)
    _per_tensor, edge_state_before = phase11b.common.state_hashes(edge.model)
    ue_state_before = {
        str(index): phase11b.common.state_hashes(model)[1]
        for index, model in enumerate(ue_models)
    }
    generator = torch.Generator(device=device).manual_seed(20260910)
    rows: list[dict[str, Any]] = []
    for action_id in (50, 71):
        profile = edge.registry.resolve(action_id)
        stream_id = f"live-candidate-cuda-a{action_id}"
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
            c2, metadata = _decode_for_equivalence(
                edge, bytes(prepared.wire_bytes), action_id
            )
            with torch.inference_mode():
                reference = edge.reference_tail(c2, metadata)
                reference_bytes = edge.reference_tail.serialize(reference)
                reference_snapshot = edge.reference_tail.take_snapshot()
                computed = edge.runtime.process_compute(
                    bytes(prepared.wire_bytes),
                    transmitted_action_id=action_id,
                )
                published = edge.runtime.publish_cpu(computed)
            if not tree_bitwise_equal(reference, computed.work.perception):
                raise RuntimeError("live candidate perception parity failed")
            if reference_bytes != published.serialized.serialized_records:
                raise RuntimeError("live candidate serialization parity failed")
            if not torch.equal(
                reference_snapshot.original_indices,
                computed.work.original_indices,
            ):
                raise RuntimeError("live candidate p025-index parity failed")
            if not torch.equal(
                reference_snapshot.semantic_labels,
                computed.work.semantic_labels,
            ):
                raise RuntimeError("live candidate segmentation parity failed")
            rows.append(
                {
                    "action_id": action_id,
                    "profile_id": profile.profile_id,
                    "frame_id": sequence,
                    "wire_bytes": len(prepared.wire_bytes),
                    "service_record_bytes": len(
                        published.serialized.serialized_records
                    ),
                    "service_record_count": published.serialized.record_count,
                }
            )
    _per_tensor, edge_state_after = phase11b.common.state_hashes(edge.model)
    ue_state_after = {
        str(index): phase11b.common.state_hashes(model)[1]
        for index, model in enumerate(ue_models)
    }
    if edge_state_before != edge_state_after or ue_state_before != ue_state_after:
        raise RuntimeError("frozen model state changed")
    return {
        "schema": "scenesense.splitfusion_freshness_live_cuda_gate.v1",
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "actions": [50, 71],
        "frames_per_action": frames_per_action,
        "rows": rows,
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "p025_indices_bitwise_identical": True,
        "segmentation_labels_bitwise_identical": True,
        "edge_frozen_state_unchanged": True,
        "ue_frozen_states_unchanged": True,
        "edge_state_sha256": edge_state_after,
        "ue_state_sha256": ue_state_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-action", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(frames_per_action=args.frames_per_action), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

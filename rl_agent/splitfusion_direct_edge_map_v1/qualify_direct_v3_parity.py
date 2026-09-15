#!/usr/bin/env python3
"""Output-parity gate for serving the direct edge->map path from the v3 tail.

The direct architecture was validated on the frozen production edge. Moving it
onto the repaired overlapped v3 tail is only admissible if the edge's observable
output does not move at all, so this gate drives both edges with the *same real
SFD1 v2 envelopes* built from real reconstructed C2 and requires exact equality
on every registered surface:

* the perception tensor tree, bitwise;
* the serialized service-record bytes;
* the constructed service rows;
* the p025 retained indices;
* the dense segmentation label map; and
* the transport byte accounting the map update reports.

It also requires that no deferred finite verdict had to be overturned.
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
    load_frozen_perception,
)
from rl_agent.splitfusion_edge_optimization_v1.optimized_tail import tree_bitwise_equal
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
from rl_agent.splitfusion_live_dispatch_v1.envelope import pack_envelope
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    STATIC_CAMERA_MODEL_SHA256,
    STATIC_CAMERA_MOUNT_SHA256,
    FrameContextV1,
    Pose6D,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.timing import UE_STAGES, StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import ProductionSplitCodec

from .direct_v3_edge import EDGE_VARIANT, preload_direct_v3_edge

REGISTERED_ACTIONS = (15, 30, 50, 71)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _context(action: int, index: int) -> FrameContextV1:
    return FrameContextV1(
        stream_id=f"direct-v3-parity-a{action}",
        frame_id=index,
        sequence_id=index + 1,
        capture_timestamp_ns=1_800_000_000_000_000_000 + index * 100_000_000,
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


def run(frames: int, actions: tuple[int, ...]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for the parity gate")
    device = torch.device("cuda:0")
    reference_edge, reference_tail, _ledger, _models = base._preload_edge(device)
    direct_edge, direct_tail, _ledger2, _models2 = preload_direct_v3_edge(device)
    _require(direct_edge.variant == EDGE_VARIANT, "direct edge variant drift")

    model, perception_base, _binding = load_frozen_perception(device)
    dataset = build_train_dataset(perception_base)
    ranker = phase11b._load_ranker(device)
    registry = SplitActionRegistry.from_runtime_binding()
    codec = ProductionSplitCodec()
    autoencoders: dict[str, Any] = {}
    for family, _family_id, bottleneck in phase11b.FAMILIES:
        if bottleneck is None:
            continue
        item = phase11b.FROZEN_INPUTS[family]
        payload = torch.load(
            phase11b._repository_path(item["path"]), map_location="cpu",
            weights_only=False,
        )
        autoencoders[family] = phase11b._load_selected_autoencoder(
            family, bottleneck, item, payload, device
        )
        del payload

    per_action: dict[str, Any] = {}
    with torch.inference_mode():
        for action in actions:
            profile = registry.resolve(action)
            autoencoder = autoencoders.get(profile.family)
            compared = 0
            for index in range(frames):
                batch = collate_batch(perception_base, dataset, [index * 137 + 11])
                c2 = encode_front(model, batch, device)[0]
                inner = codec.encode(
                    profile, c2,
                    ranker=(None if profile.q_e4 == 0 else ranker),
                    ae_encoder=autoencoder,
                    timing=StageRecorder(UE_STAGES),
                )
                context = _context(action, index)
                frame = pack_envelope(
                    inner,
                    action_id=action,
                    sequence_id=context.sequence_id,
                    capture_timestamp_ns=context.capture_timestamp_ns,
                    frame_context=context,
                )
                reference = reference_edge.process(frame, transmitted_action_id=action)
                reference_snapshot = reference_tail.take_snapshot()
                observed = direct_edge.process(frame, transmitted_action_id=action)
                observed_snapshot = direct_tail.take_snapshot()
                where = f"action {action} frame {index}"
                _require(
                    tree_bitwise_equal(reference.perception, observed.perception),
                    f"{where}: perception drift",
                )
                _require(
                    reference.serialized_output == observed.serialized_output,
                    f"{where}: serialized service-record drift",
                )
                _require(
                    list(reference_snapshot.records or ())
                    == list(observed_snapshot.records or ()),
                    f"{where}: service row drift",
                )
                _require(
                    torch.equal(
                        reference_snapshot.original_indices,
                        observed_snapshot.original_indices,
                    ),
                    f"{where}: p025 index drift",
                )
                _require(
                    torch.equal(
                        reference_snapshot.semantic_labels,
                        observed_snapshot.semantic_labels,
                    ),
                    f"{where}: segmentation label drift",
                )
                _require(
                    reference.scientific_inner_payload_bytes
                    == observed.scientific_inner_payload_bytes
                    and reference.framing_control_overhead_bytes
                    == observed.framing_control_overhead_bytes
                    and reference.total_received_bytes == observed.total_received_bytes,
                    f"{where}: transport byte accounting drift",
                )
                compared += 1
            per_action[str(action)] = {
                "profile_id": profile.profile_id,
                "family": profile.family,
                "frames_compared": compared,
                "perception_bitwise_identical": True,
                "service_records_byte_identical": True,
                "service_rows_identical": True,
                "p025_indices_bitwise_identical": True,
                "segmentation_labels_bitwise_identical": True,
                "transport_byte_accounting_identical": True,
            }

    corrections = int(direct_edge.asynchronous_verdict_corrections)
    return {
        "schema": "scenesense.splitfusion_direct_v3_parity.v1",
        "device": torch.cuda.get_device_name(device),
        "edge_variant": direct_edge.variant,
        "reference": "rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime._preload_edge",
        "frames_per_action": frames,
        "actions": list(actions),
        "per_action": per_action,
        "asynchronous_verdict_corrections": corrections,
        "passed": corrections == 0
        and all(
            entry["frames_compared"] == frames for entry in per_action.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--actions", default=",".join(str(a) for a in REGISTERED_ACTIONS))
    args = parser.parse_args()
    report = run(
        args.frames, tuple(int(value) for value in args.actions.split(","))
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate the canonical offline fixture documents.

These fixtures stand in for real wire documents because this branch cannot
import the frozen production protocol module. They are shaped to match
``protocol.build_object_map_update`` / ``build_map_feedback`` at the pinned
digests recorded in ``reference_pins.py``.

Pure: writes JSON files and nothing else. No socket, no CARLA, no service.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent

RUN_ID = "coop_fixture_run_v1"
CELL_ID = "coop_fixture_cell_a20_favorable_stable"
SESSION_ID = "two-ue-live-action50-v1"
STREAM_A = "two-ue/ue-a"
STREAM_B = "two-ue/ue-b"
ACTION_ID = 50
PROFILE_ID = "split_ae64_uint4_q5000"
DECODER_IDENTITY = "repaired_v3_tail_ae64_uint4"

# Unix wall nanoseconds: the production capture clock domain.
CAPTURE_NS_A = 1_789_000_000_123_456_789
CAPTURE_NS_B = 1_789_000_000_223_456_789
SERVICE_S = 0.5
ACK_S = 1.0

# CARLA simulation nanoseconds: the HTTP demo capture clock domain. Deliberately
# a different magnitude so a test can see that the two are not interchangeable.
SIM_CAPTURE_NS_A = 612_400_000_000
SIM_CAPTURE_NS_B = 612_500_000_000


def p025_record(
    *, stream_id: str, frame_id: int, capture_ns: int, index: int, world_x: float
) -> dict[str, Any]:
    """One compact p025 object record, in the deployed field layout."""

    return {
        "stream_id": stream_id,
        "capture_timestamp_ns": capture_ns,
        "sample_id": f"{stream_id}:{frame_id}",
        "frame_id": frame_id,
        "prediction_index": index,
        "class_name": "vehicle",
        "internal_class": 1,
        "score": 0.8125,
        "world_x": world_x,
        "world_y": 28.25,
        "world_z": 0.5,
        "local_x": 11.5,
        "local_y": 0.75,
        "local_z": 0.0,
        "size_x": 4.6,
        "size_y": 2.0,
        "size_z": 1.55,
        "yaw_sin": 0.0,
        "yaw_cos": 1.0,
        "parked_score": 0.0625,
        "radar_support_score": 0.75,
        "center_x_px": 384.0,
        "center_y_px": 224.0,
        "bbox_x0": 352.0,
        "bbox_y0": 200.0,
        "bbox_x1": 416.0,
        "bbox_y1": 248.0,
        "fpn_level": "p3",
        "level_index": 0,
        "point_index": 1024 + index,
        "candidate_identity": f"{frame_id}:0:{1024 + index}",
        "physical_ray_x_px": 384.0,
        "physical_ray_y_px": 224.0,
        "actor_forward_depth_m": 11.5,
        "depth_bin": 4,
        "depth_residual": 0.125,
    }


def scalar_count(value: Any) -> int:
    """Leaf count, matching protocol.assert_finite_tree's accounting."""

    if isinstance(value, Mapping):
        return sum(scalar_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(scalar_count(item) for item in value)
    if isinstance(value, float):
        assert math.isfinite(value)
    return 1


def direct_update(
    *, stream_id: str, frame_id: int, sequence_id: int, capture_ns: int, world_x: float
) -> dict[str, Any]:
    records = [
        p025_record(
            stream_id=stream_id,
            frame_id=frame_id,
            capture_ns=capture_ns,
            index=0,
            world_x=world_x,
        )
    ]
    capture_s = capture_ns / 1_000_000_000.0
    return {
        "schema": "splitfusion_direct_object_map_update.v1",
        "protocol_version": 1,
        "run_id": RUN_ID,
        "cell_id": CELL_ID,
        "stream_id": stream_id,
        "frame_id": frame_id,
        "sequence_id": sequence_id,
        "action_id": ACTION_ID,
        "profile_id": PROFILE_ID,
        "decoder_identity": DECODER_IDENTITY,
        "capture_timestamp_ns": capture_ns,
        "capture_timestamp": capture_s,
        # The live direct runtime passes 0.0: the production document carries no
        # usable CARLA simulation clock.
        "carla_timestamp": 0.0,
        "service_deadline_at": capture_s + SERVICE_S,
        "ack_timeout_at": capture_s + ACK_S,
        "record_count": len(records),
        "record_scalar_count": scalar_count(records),
        "records": records,
        "edge_timing": {
            "tail_complete_wall_s": capture_s + 0.181,
            "reassembly_complete_wall_s": capture_s + 0.152,
            "admission_wall_s": capture_s + 0.156,
            "compute_start_wall_s": capture_s + 0.158,
            "compute_finish_wall_s": capture_s + 0.179,
            "publication_ticket_ready_wall_s": capture_s + 0.182,
            "publish_start_wall_s": capture_s + 0.183,
            "serialization_start_wall_s": capture_s + 0.1832,
        },
        "segmentation": {
            "available": True,
            "evidence": {"persisted": True, "hash_verified": True},
            "installation_status": "EDGE_MOUNT_PERSISTED",
        },
    }


def map_feedback(update: Mapping[str, Any], *, install_offset_s: float) -> dict[str, Any]:
    capture_s = int(update["capture_timestamp_ns"]) / 1_000_000_000.0
    install_at = capture_s + install_offset_s
    return {
        "schema": "splitfusion_direct_map_feedback.v1",
        "protocol_version": 1,
        "source": "EDGE_SPATIAL_MAP",
        "run_id": update["run_id"],
        "cell_id": update["cell_id"],
        "stream_id": update["stream_id"],
        "frame_id": update["frame_id"],
        "capture_id": f"{update['stream_id']}:{int(update['frame_id'])}",
        "action_id": update["action_id"],
        "profile_id": update["profile_id"],
        "capture_timestamp_ns": update["capture_timestamp_ns"],
        "capture_timestamp": capture_s,
        "service_deadline_at": update["service_deadline_at"],
        "ack_timeout_at": update["ack_timeout_at"],
        "outcome": "RESULT_INSTALLED",
        "agent_credit": "CREDIT_INSTALLED_ON_TIME",
        "terminal": True,
        "accepted": True,
        "install_timestamp": install_at,
        "map_ingest_at": install_at - 0.0015,
        "feedback_emit_at": install_at + 0.0008,
        "map_age_at_install_ms": install_offset_s * 1000.0,
        "rejection_reason": "",
        "direct_update_bytes": 2048,
        "direct_update_datagrams": 1,
        "superseded_by_frame_id": "",
        "edge_timing": dict(update["edge_timing"]),
    }


def pose_sidecar(update: Mapping[str, Any], *, x: float, y: float, yaw: float) -> dict[str, Any]:
    return {
        "schema": "scenesense.coop_pose_sidecar.v1",
        "sidecar_version": 1,
        "run_id": update["run_id"],
        "cell_id": update["cell_id"],
        "stream_id": update["stream_id"],
        "frame_id": update["frame_id"],
        "capture_timestamp_ns": update["capture_timestamp_ns"],
        "ego_world": {
            "x": x, "y": y, "z": 0.6,
            "pitch": 0.0, "yaw": yaw, "roll": 0.0,
        },
        "horizontal_fov_deg": 120.0,
        "pose_provenance": "SFD1_FRAME_CONTEXT_V1_EGO_WORLD",
    }


def http_live_view_update(
    *, stream_id: str, ue_id: str, frame_id: int, capture_ns: int, world_x: float,
    ego_x: float, ego_y: float, ego_yaw: float,
) -> dict[str, Any]:
    records = [
        p025_record(
            stream_id=stream_id,
            frame_id=frame_id,
            capture_ns=capture_ns,
            index=0,
            world_x=world_x,
        )
    ]
    return {
        "schema": "scenesense.multi_ue_live_view_update.v1",
        "ue_id": ue_id,
        "session_id": SESSION_ID,
        # The demonstration's capture clock is the CARLA simulation clock.
        "clock_domain": "carla_simulation_ns",
        "horizontal_fov_deg": 120.0,
        "ego_pose": {
            "actor_id": 412,
            "location": {"x": ego_x, "y": ego_y, "z": 0.6},
            "rotation": {"pitch": 0.0, "yaw": ego_yaw, "roll": 0.0},
        },
        "payload": {
            "schema": "splitfusion_edge_result.v2",
            "stream_id": stream_id,
            "frame_id": frame_id,
            "capture_timestamp_ns": capture_ns,
            "action_id": ACTION_ID,
            "profile_id": PROFILE_ID,
            "object_map_update": {
                "schema": "splitfusion_object_map_update.v1",
                "stream_id": stream_id,
                "frame_id": frame_id,
                "capture_timestamp_ns": capture_ns,
                "action_id": ACTION_ID,
                "records": records,
            },
        },
    }


def identity_bindings() -> dict[str, Any]:
    return {
        "schema": "scenesense.coop_identity_binding.v1",
        "bindings": [
            {
                "run_id": RUN_ID,
                "cell_id": CELL_ID,
                "stream_id": STREAM_A,
                "ue_id": "ue-a",
                "session_id": SESSION_ID,
            },
            {
                "run_id": RUN_ID,
                "cell_id": CELL_ID,
                "stream_id": STREAM_B,
                "ue_id": "ue-b",
                "session_id": SESSION_ID,
            },
        ],
    }


def demo_identity_bindings() -> dict[str, Any]:
    """HTTP demo bindings: no run/cell concept, declared explicitly as null."""

    return {
        "schema": "scenesense.coop_identity_binding.v1",
        "bindings": [
            {
                "run_id": None,
                "cell_id": None,
                "stream_id": STREAM_A,
                "ue_id": "ue-a",
                "session_id": SESSION_ID,
            },
            {
                "run_id": None,
                "cell_id": None,
                "stream_id": STREAM_B,
                "ue_id": "ue-b",
                "session_id": SESSION_ID,
            },
        ],
    }


def write(name: str, document: Any) -> Path:
    path = HERE / name
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def main() -> int:
    update_a = direct_update(
        stream_id=STREAM_A, frame_id=4096, sequence_id=2048,
        capture_ns=CAPTURE_NS_A, world_x=11.5,
    )
    update_b = direct_update(
        stream_id=STREAM_B, frame_id=4098, sequence_id=2049,
        capture_ns=CAPTURE_NS_B, world_x=12.25,
    )
    written = [
        write("direct_object_map_update_a_v1.json", update_a),
        write("direct_object_map_update_b_v1.json", update_b),
        write("direct_map_feedback_a_v1.json", map_feedback(update_a, install_offset_s=0.194)),
        write("pose_sidecar_a_v1.json", pose_sidecar(update_a, x=-4.0, y=28.1, yaw=0.16)),
        write("identity_bindings_direct_v1.json", identity_bindings()),
        write("identity_bindings_http_demo_v1.json", demo_identity_bindings()),
        write(
            "http_live_view_update_a_v1.json",
            http_live_view_update(
                stream_id=STREAM_A, ue_id="ue-a", frame_id=4096,
                capture_ns=SIM_CAPTURE_NS_A, world_x=11.5,
                ego_x=-4.0, ego_y=28.1, ego_yaw=0.16,
            ),
        ),
        write(
            "http_live_view_update_b_v1.json",
            http_live_view_update(
                stream_id=STREAM_B, ue_id="ue-b", frame_id=4098,
                capture_ns=SIM_CAPTURE_NS_B, world_x=12.25,
                ego_x=-19.0, ego_y=28.05, ego_yaw=0.2,
            ),
        ),
    ]
    for path in written:
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

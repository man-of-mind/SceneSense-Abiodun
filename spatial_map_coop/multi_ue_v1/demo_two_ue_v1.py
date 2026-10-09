#!/usr/bin/env python3
"""Deterministic, CARLA-free two-UE ingress/association demonstration."""

from __future__ import annotations

import json

from .association import AssociationPolicy
from .service import MultiUESpatialMapService


def edge_result(stream_id: str, frame_id: int, capture_ns: int, x: float, score: float):
    record = {
        "stream_id": stream_id,
        "frame_id": frame_id,
        "capture_timestamp_ns": capture_ns,
        "candidate_identity": f"{frame_id}:vehicle:0",
        "class_name": "vehicle",
        "score": score,
        "world_x": x,
        "world_y": 8.0,
        "world_z": 0.8,
        "size_x": 4.6,
        "size_y": 1.9,
        "size_z": 1.5,
        "yaw_sin": 0.0,
        "yaw_cos": 1.0,
    }
    update = {
        "schema": "splitfusion_object_map_update.v1",
        "stream_id": stream_id,
        "frame_id": frame_id,
        "capture_timestamp_ns": capture_ns,
        "action_id": 71,
        "records": [record],
    }
    return {
        "schema": "splitfusion_edge_result.v2",
        "stream_id": stream_id,
        "frame_id": frame_id,
        "capture_timestamp_ns": capture_ns,
        "action_id": 71,
        "object_map_update": update,
    }


def main() -> None:
    policy = AssociationPolicy(
        maximum_observation_age_ns=500_000_000,
        alignment_tolerance_ns=100_000_000,
        maximum_pair_time_delta_ns=100_000_000,
        maximum_xy_distance_m=4.0,
        maximum_relative_size_difference=0.65,
        track_match_distance_m=6.0,
        track_stale_after_ns=4_000_000_000,
        minimum_confidence=0.0,
    )
    service = MultiUESpatialMapService(policy)
    service.ingest_splitfusion(
        edge_result("ego-a", 100, 1_000_000_000, 20.0, 0.91),
        ue_id="ue-a",
        session_id="demo-drive",
        received_timestamp_ns=1_025_000_000,
    )
    service.ingest_splitfusion(
        edge_result("ego-b", 200, 1_040_000_000, 20.4, 0.77),
        ue_id="ue-b",
        session_id="demo-drive",
        received_timestamp_ns=1_060_000_000,
    )
    document = service.snapshot(
        clock_domain="unix_wall_ns",
        snapshot_timestamp_ns=1_080_000_000,
    ).as_dict()
    if len(document["tracks"]) != 1:
        raise RuntimeError("two corresponding UE observations did not form one track")
    track = document["tracks"][0]
    if track["selected_ue_id"] != "ue-b" or len(track["contributing_sources"]) != 2:
        raise RuntimeError("freshest-source selection or provenance binding failed")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

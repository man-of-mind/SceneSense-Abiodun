#!/usr/bin/env python3
"""Synthetic spatial-map snapshots matching the live /api/spatial_map/latest schema, for CARLA-free
offline development + verification of the Stage-2 (color-by-source) rendering.

Produces (snapshot, static_geometry) pairs. Kept intentionally simple/deterministic (no RNG) so figures
are reproducible.
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def _obj(src: str, otype: str, x: float, y: float, yaw: float = 0.0,
         L: float = 4.5, W: float = 1.9, score: float = 0.8) -> Dict:
    return {
        "source_stream_id": src,
        "type": otype,
        "score": score,
        "location": {"x": float(x), "y": float(y), "z": 0.0},
        "dimensions": {"length": float(L), "width": float(W), "height": 1.6},
        "yaw_deg": float(yaw),
    }


def two_ego_scene() -> Tuple[Dict, Dict]:
    """Two egos on the same straight road (B trailing A), several NPCs, and ONE object detected by both
    cars ~1.2 m apart (the pre-fusion duplication that motivates Stage 3)."""
    ego_a = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    radius_m, padding_m = 40.0, 10.0

    # NOTE: vehicle yaw_deg values are deliberately WRONG/slanted here to mimic the model's unreliable
    # yaw; road-snap heading should straighten them onto the axis-aligned roads (0 deg on the y=0 road).
    objs: List[Dict] = []
    # car A's detections (ahead of A), all on/near the horizontal road (true orientation ~0 deg)
    objs += [
        _obj("fusion_ego", "Vehicle", 18.0, 1.5, 38.0),
        _obj("fusion_ego", "Vehicle", 30.0, -3.5, -47.0),
        _obj("fusion_ego", "Pedestrian", 12.0, -6.0, 0.0, L=0.6, W=0.6),
        _obj("fusion_ego", "Vehicle", -14.0, 0.0, 22.0),  # this is car B itself, seen by A
    ]
    # car B (trailing A at x=-15) — overlapping view of the shared vehicle at ~ (18,1.5)
    objs += [
        _obj("fusion_ego_b", "Vehicle", 19.2, 1.0, 55.0),  # SAME object as A's (18,1.5) -> duplicate
        _obj("fusion_ego_b", "Vehicle", 2.0, 4.0, -30.0),  # only B sees this one
        _obj("fusion_ego_b", "Pedestrian", 6.0, -4.5, 0.0, L=0.6, W=0.6),
    ]

    anchors = [
        {"location": {"x": 25.0, "y": 10.0}, "active": True, "focus": False},
        {"location": {"x": -5.0, "y": -12.0}, "active": False, "focus": False},
    ]

    snap = {
        "frame_id": 1000,
        "status": "active",
        "raw_spatial_map_objects": objs,
        "spatial_map_objects": [],
        "traffic_light_anchors": anchors,
        "active_streams": [
            {"stream_id": "fusion_ego", "object_count": 4},
            {"stream_id": "fusion_ego_b", "object_count": 3},
        ],
        "metadata": {
            "coordinate_system": "global_carla_world",
            "focus_view": {
                "mode": "follow_stream",
                "follow_stream_id": "fusion_ego",
                "radius_m": radius_m,
                "padding_m": padding_m,
                "ego_pose": ego_a,
                "bounds": {
                    "x_min": ego_a["x"] - radius_m - padding_m,
                    "x_max": ego_a["x"] + radius_m + padding_m,
                    "y_min": ego_a["y"] - radius_m - padding_m,
                    "y_max": ego_a["y"] + radius_m + padding_m,
                },
            },
        },
    }

    # minimal static backdrop: two road centrelines + one building footprint
    static = {
        "map_name": "SYNTHETIC",
        "roads": [
            [[-60.0, 0.0], [60.0, 0.0]],
            [[0.0, -60.0], [0.0, 60.0]],
        ],
        "buildings": [
            [[8.0, 8.0], [22.0, 8.0], [22.0, 20.0], [8.0, 20.0]],
        ],
    }
    return snap, static


def occlusion_scene():
    """Stage-3 groundwork scene with KNOWN ground truth.
    Car A at (0,0) and car B 15 m behind at (-15,0), both facing +x. A pedestrian at (25,3) is visible to
    A; a truck at (18,2) sits between car B and the pedestrian, so B's view of the pedestrian is occluded.
    Both cars see the truck. Returns (streams, ground_truth):
      streams: [{stream_id, pose{x,y,yaw_deg}, fov_deg, range_m, objects:[{type,x,y,length,width,yaw_deg}]}]
      ground_truth: list of true occlusions {class_name, x, y, seen_by, occluded_from}
    """
    def o(t, x, y, L, W, yaw=0.0):
        return {"type": t, "x": x, "y": y, "length": L, "width": W, "yaw_deg": yaw}

    truck = o("Vehicle", 18.0, 2.0, 6.0, 2.6)
    streams = [
        {"stream_id": "fusion_ego", "pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
         "fov_deg": 90.0, "range_m": 60.0,
         "objects": [dict(truck), o("Pedestrian", 25.0, 3.0, 0.6, 0.6)]},
        {"stream_id": "fusion_ego_b", "pose": {"x": -15.0, "y": 0.0, "yaw_deg": 0.0},
         "fov_deg": 90.0, "range_m": 60.0,
         "objects": [dict(truck)]},  # pedestrian at (25,3) occluded by the truck -> not detected by B
    ]
    ground_truth = [{"class_name": "Pedestrian", "x": 25.0, "y": 3.0,
                     "seen_by": "fusion_ego", "occluded_from": "fusion_ego_b"}]
    return streams, ground_truth


if __name__ == "__main__":
    import json
    snap, static = two_ego_scene()
    print("streams:", [s["stream_id"] for s in snap["active_streams"]])
    print("objects:", len(snap["raw_spatial_map_objects"]))
    print(json.dumps(snap["metadata"]["focus_view"]["ego_pose"]))

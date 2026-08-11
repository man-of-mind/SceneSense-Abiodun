#!/usr/bin/env python3
"""Policy-corpus entry point: shared fusion collector plus pedestrian GT.

The large, validated collection pipeline remains in
``uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py``.
This module deliberately overlays only its ground-truth row builder so the
shared source is unchanged and the policy corpus does not maintain a divergent
copy of the real-time perception path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NEU_COLLAB_ROOT = REPO_ROOT.parent
# The inherited collector conditionally inserts neu_collab ahead of abiodun if
# only the latter is already present. Pre-register both in the intended order;
# otherwise the stale parent data-collect module lacks the remote_host API.
for _path in (str(REPO_ROOT), str(NEU_COLLAB_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(NEU_COLLAB_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from uplink_only_spatial_map_pipeline import (  # noqa: E402
    carla_fusion_staleness_scenario_uplink_only as base,
)

if Path(base.od_collect.__file__).resolve().parent != REPO_ROOT:
    raise RuntimeError(
        "stale split-inference module resolved outside abiodun: "
        f"{base.od_collect.__file__}. Do not export PYTHONPATH."
    )


_BUILD_VEHICLE_ROWS = base.build_vehicle_ground_truth_rows
_BUILD_FUSION_METRICS_ROW = base.build_fusion_metrics_row


def build_policy_corpus_metrics_row(*args: object, **kwargs: object) -> Dict[str, object]:
    """Expose the capture/render timing already measured by the shared loop."""

    row = _BUILD_FUSION_METRICS_ROW(*args, **kwargs)
    front_stats = kwargs.get("front_stats")
    if not isinstance(front_stats, dict):
        front_stats = {}
    row["camera_frame_wait_ms"] = base._safe_float(
        front_stats.get("camera_frame_wait_ms"), float("nan")
    )
    return row


def _build_pedestrian_ground_truth_rows(
    *,
    world: "base.carla.World",
    frame_id: int,
    elapsed_s: float,
    carla_timestamp: float,
    camera_transform: "base.carla.Transform",
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
    camera_width: int,
    camera_height: int,
) -> List[Dict[str, object]]:
    """Return walker rows using the existing vehicle schema and conventions."""

    camera_location = camera_transform.location
    rows: List[Dict[str, object]] = []
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        try:
            transform = actor.get_transform()
            bbox = actor.bounding_box
            projection = base._project_actor_bbox_to_image(
                actor,
                camera_inverse_matrix=camera_inverse_matrix,
                intrinsics=intrinsics,
                camera_width=int(camera_width),
                camera_height=int(camera_height),
            )
        except RuntimeError:
            continue

        center_world = np.asarray(projection["center_world"], dtype=np.float64)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = base._bbox_xyxy_values(
            projection["bbox_xyxy"]
        )
        distance_m = math.sqrt(
            (float(center_world[0]) - float(camera_location.x)) ** 2
            + (float(center_world[1]) - float(camera_location.y)) ** 2
            + (float(center_world[2]) - float(camera_location.z)) ** 2
        )
        try:
            role_name = str(actor.attributes.get("role_name", ""))
        except Exception:
            role_name = ""
        rows.append(
            {
                "elapsed_s": float(elapsed_s),
                "frame_id": int(frame_id),
                "carla_timestamp": float(carla_timestamp),
                "actor_id": int(actor.id),
                "type_id": str(getattr(actor, "type_id", "")),
                "role_name": role_name,
                "class_name": "pedestrian",
                # Preserve the established schema: world_* is bbox center.
                "world_x": float(center_world[0]),
                "world_y": float(center_world[1]),
                "world_z": float(center_world[2]),
                # Matching/replay must use actor origin, as used during training.
                "origin_x": float(transform.location.x),
                "origin_y": float(transform.location.y),
                "origin_z": float(transform.location.z),
                "yaw_deg": float(transform.rotation.yaw),
                "length_m": float(bbox.extent.x) * 2.0,
                "width_m": float(bbox.extent.y) * 2.0,
                "height_m": float(bbox.extent.z) * 2.0,
                "distance_m": float(distance_m),
                "in_camera_frustum": int(bool(projection["in_camera_frustum"])),
                "projected_x": float(projection["projected_x"]),
                "projected_y": float(projection["projected_y"]),
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
            }
        )
    return rows


def build_object_ground_truth_rows(
    *,
    world: "base.carla.World",
    frame_id: int,
    elapsed_s: float,
    carla_timestamp: float,
    camera_transform: "base.carla.Transform",
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
    camera_width: int,
    camera_height: int,
    exclude_actor_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, object]]:
    """Append pedestrian truth to the unchanged vehicle-truth implementation."""

    rows = _BUILD_VEHICLE_ROWS(
        world=world,
        frame_id=frame_id,
        elapsed_s=elapsed_s,
        carla_timestamp=carla_timestamp,
        camera_transform=camera_transform,
        camera_inverse_matrix=camera_inverse_matrix,
        intrinsics=intrinsics,
        camera_width=camera_width,
        camera_height=camera_height,
        exclude_actor_ids=exclude_actor_ids,
    )
    rows.extend(
        _build_pedestrian_ground_truth_rows(
            world=world,
            frame_id=frame_id,
            elapsed_s=elapsed_s,
            carla_timestamp=carla_timestamp,
            camera_transform=camera_transform,
            camera_inverse_matrix=camera_inverse_matrix,
            intrinsics=intrinsics,
            camera_width=camera_width,
            camera_height=camera_height,
        )
    )
    return rows


def main() -> None:
    base.build_vehicle_ground_truth_rows = build_object_ground_truth_rows
    if "camera_frame_wait_ms" not in base.FUSION_METRICS_FIELDS:
        base.FUSION_METRICS_FIELDS = (*base.FUSION_METRICS_FIELDS, "camera_frame_wait_ms")
    base.build_fusion_metrics_row = build_policy_corpus_metrics_row
    # Make the inherited manifest name the actual collection entry point.
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()

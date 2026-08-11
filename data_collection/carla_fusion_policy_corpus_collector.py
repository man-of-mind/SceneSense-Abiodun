#!/usr/bin/env python3
"""Policy-corpus entry point: shared fusion collector plus pedestrian GT.

The large, validated collection pipeline remains in
``uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py``.
This module deliberately overlays only its ground-truth row builder so the
shared source is unchanged and the policy corpus does not maintain a divergent
copy of the real-time perception path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    object_reg_channels,
)

if Path(base.od_collect.__file__).resolve().parent != REPO_ROOT:
    raise RuntimeError(
        "stale split-inference module resolved outside abiodun: "
        f"{base.od_collect.__file__}. Do not export PYTHONPATH."
    )


_BUILD_VEHICLE_ROWS = base.build_vehicle_ground_truth_rows
_BUILD_FUSION_METRICS_ROW = base.build_fusion_metrics_row
_DECODE_OBJECTS = base.decode_objects
_RUN_BACK_HALF = base.FusionRemoteInferenceWorker._run_back_half
_SPAWN_LEAD_TARGET = base._spawn_lead_target
_SPAWN_CONTROLLED_TARGET = base._spawn_controlled_target
_WRITE_MANIFEST = base.FusionRunLogger.write_manifest
_DECODE_DIAGNOSTICS = threading.local()
_DECODE_BY_FRAME: Dict[int, Dict[str, int]] = {}
_DECODE_BY_FRAME_LOCK = threading.Lock()


@dataclass(frozen=True)
class ControlledPedestrianOverlay:
    """Track-B-only scene controls stripped before the shared parser runs."""

    crowd_count: int = 0
    crowd_min_spawned: int = 0
    crowd_depth_min_m: float = 14.0
    crowd_depth_max_m: float = 22.0
    crowd_depth_step_m: float = 2.0
    crowd_lateral_spacing_m: float = 0.85
    crowd_speed_mps: float = 0.0
    target_start_lateral_m: float = 0.0
    horizontal_fov_deg: float = 100.0
    headline_range_m: float = 25.0


_PEDESTRIAN_OVERLAY = ControlledPedestrianOverlay()
_OVERLAY_ACTORS: List[object] = []
_CONTROLLED_TARGET_INFO: Optional[Dict[str, object]] = None


def _parse_overlay_args(
    argv: Sequence[str],
) -> Tuple[ControlledPedestrianOverlay, List[str]]:
    """Parse policy-overlay flags while preserving all inherited CLI flags."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--controlled-pedestrian-crowd-count", type=int, default=0)
    parser.add_argument("--controlled-pedestrian-crowd-min-spawned", type=int, default=0)
    parser.add_argument("--controlled-pedestrian-crowd-depth-min-m", type=float, default=14.0)
    parser.add_argument("--controlled-pedestrian-crowd-depth-max-m", type=float, default=22.0)
    parser.add_argument("--controlled-pedestrian-crowd-depth-step-m", type=float, default=2.0)
    parser.add_argument("--controlled-pedestrian-crowd-lateral-spacing-m", type=float, default=0.85)
    parser.add_argument("--controlled-pedestrian-crowd-speed-mps", type=float, default=0.0)
    parser.add_argument("--controlled-pedestrian-target-start-lateral-m", type=float, default=0.0)
    parser.add_argument("--controlled-pedestrian-horizontal-fov-deg", type=float, default=100.0)
    parser.add_argument("--controlled-pedestrian-headline-range-m", type=float, default=25.0)
    parsed, remaining = parser.parse_known_args(list(argv))
    overlay = ControlledPedestrianOverlay(
        crowd_count=int(parsed.controlled_pedestrian_crowd_count),
        crowd_min_spawned=int(parsed.controlled_pedestrian_crowd_min_spawned),
        crowd_depth_min_m=float(parsed.controlled_pedestrian_crowd_depth_min_m),
        crowd_depth_max_m=float(parsed.controlled_pedestrian_crowd_depth_max_m),
        crowd_depth_step_m=float(parsed.controlled_pedestrian_crowd_depth_step_m),
        crowd_lateral_spacing_m=float(parsed.controlled_pedestrian_crowd_lateral_spacing_m),
        crowd_speed_mps=float(parsed.controlled_pedestrian_crowd_speed_mps),
        target_start_lateral_m=float(parsed.controlled_pedestrian_target_start_lateral_m),
        horizontal_fov_deg=float(parsed.controlled_pedestrian_horizontal_fov_deg),
        headline_range_m=float(parsed.controlled_pedestrian_headline_range_m),
    )
    if overlay.crowd_count < 0:
        raise ValueError("controlled pedestrian crowd count must be non-negative")
    if not 0 <= overlay.crowd_min_spawned <= overlay.crowd_count:
        raise ValueError("controlled pedestrian crowd minimum must be within [0, crowd count]")
    if not 0.0 < overlay.crowd_depth_min_m <= overlay.crowd_depth_max_m:
        raise ValueError("controlled pedestrian crowd depth range is invalid")
    if overlay.crowd_depth_step_m <= 0.0 or overlay.crowd_lateral_spacing_m <= 0.0:
        raise ValueError("controlled pedestrian crowd spacing must be positive")
    if overlay.crowd_speed_mps < 0.0:
        raise ValueError("controlled pedestrian crowd speed must be non-negative")
    if not 1.0 < overlay.horizontal_fov_deg < 179.0:
        raise ValueError("controlled pedestrian horizontal FOV must be within (1, 179) degrees")
    if overlay.headline_range_m <= overlay.crowd_depth_min_m:
        raise ValueError("controlled pedestrian headline range must exceed minimum depth")
    return overlay, remaining


def _compose_camera_world_matrix(
    anchor_transform: object,
    relative_camera_transform: object,
) -> np.ndarray:
    """Compose an attached sensor's relative transform with its actor world pose."""

    anchor_matrix = np.asarray(anchor_transform.get_matrix(), dtype=np.float64)
    relative_matrix = np.asarray(relative_camera_transform.get_matrix(), dtype=np.float64)
    if anchor_matrix.shape != (4, 4) or relative_matrix.shape != (4, 4):
        raise ValueError("CARLA transform matrices must be 4x4")
    return anchor_matrix @ relative_matrix


def _resolve_anchor_transform(world: object, anchor_location: object) -> object:
    """Resolve the ego actor whose pose owns the relative camera transform."""

    candidates = list(world.get_actors().filter("vehicle.*"))
    ranked = []
    for actor in candidates:
        try:
            location = actor.get_location()
            distance = math.sqrt(
                (float(location.x) - float(anchor_location.x)) ** 2
                + (float(location.y) - float(anchor_location.y)) ** 2
                + (float(location.z) - float(anchor_location.z)) ** 2
            )
            role_name = str(actor.attributes.get("role_name", ""))
            ranked.append((distance, role_name != "scenesense_fusion_ego", int(actor.id), actor))
        except RuntimeError:
            continue
    if not ranked:
        raise RuntimeError("cannot compose controlled-walker world pose: no vehicle anchor")
    ranked.sort(key=lambda item: item[:3])
    distance, _, _, actor = ranked[0]
    if float(distance) > 2.0:
        raise RuntimeError(
            "cannot compose controlled-walker world pose: nearest vehicle is "
            f"{float(distance):.2f} m from the sensor anchor"
        )
    return actor.get_transform()


def _pedestrian_crowd_offsets(overlay: ControlledPedestrianOverlay) -> List[Tuple[float, float]]:
    """Return deterministic close, in-frustum (forward, lateral) crowd slots."""

    depth_values = np.arange(
        overlay.crowd_depth_min_m,
        overlay.crowd_depth_max_m + 0.5 * overlay.crowd_depth_step_m,
        overlay.crowd_depth_step_m,
        dtype=np.float64,
    )
    offsets: List[Tuple[float, float]] = []
    fov_half_rad = math.radians(0.40 * overlay.horizontal_fov_deg)
    range_margin_m = 0.75
    for depth in depth_values:
        range_limit = math.sqrt(
            max(0.0, (overlay.headline_range_m - range_margin_m) ** 2 - float(depth) ** 2)
        )
        frustum_limit = float(depth) * math.tan(fov_half_rad)
        lateral_limit = max(0.0, min(range_limit, frustum_limit) - 0.5)
        slot_count = int(math.floor((2.0 * lateral_limit) / overlay.crowd_lateral_spacing_m)) + 1
        if slot_count <= 0:
            continue
        row = np.linspace(-lateral_limit, lateral_limit, slot_count, dtype=np.float64)
        if len(row) > 1 and float(row[1] - row[0]) < overlay.crowd_lateral_spacing_m - 1e-9:
            row = row[:-1]
        offsets.extend((float(depth), float(lateral)) for lateral in row)
    return offsets


def _destroy_overlay_actors() -> None:
    """Best-effort cleanup for crowd actors not owned by the shared actor list."""

    global _OVERLAY_ACTORS
    actors, _OVERLAY_ACTORS = list(_OVERLAY_ACTORS), []
    if not actors:
        return
    try:
        base.pole_client._destroy_actors(actors)
    except Exception:
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass


def spawn_lead_target_with_synchronized_exact_start(*args: object, **kwargs: object):
    """Arm ego velocity before the base exact-convoy preflight tick.

    The shared helper arms the lead before returning, while the caller normally
    arms the ego immediately afterward. On a fast-rendering server one
    synchronous frame can occur between those operations, creating an artificial
    ``speed/fps`` gap jump. Pre-arming the ego here makes the first tick paired;
    the caller's repeated idempotent setup remains unchanged.
    """

    actor = _SPAWN_LEAD_TARGET(*args, **kwargs)
    if str(kwargs.get("motion_control", "")) == "exact" and str(kwargs.get("kind", "")) == "vehicle":
        ego_vehicle = kwargs["ego_vehicle"]
        speed_mps = float(kwargs["speed_mps"])
        ego_vehicle.set_autopilot(False)
        ego_vehicle.set_simulate_physics(True)
        ego_vehicle.apply_control(
            base.carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
        )
        ego_vehicle.enable_constant_velocity(
            base.carla.Vector3D(x=speed_mps, y=0.0, z=0.0)
        )
    return actor


def _fresh_walker_blueprint(world: object, index: int, role_name: str) -> object:
    blueprints = sorted(
        list(world.get_blueprint_library().filter("walker.pedestrian.*")),
        key=lambda blueprint: str(blueprint.id),
    )
    if not blueprints:
        raise RuntimeError("no walker blueprints are available")
    source = blueprints[int(index) % len(blueprints)]
    try:
        blueprint = world.get_blueprint_library().find(source.id)
    except Exception:
        blueprint = source
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    return blueprint


def _world_location_from_camera_frame(
    camera_world_matrix: np.ndarray,
    *,
    forward_m: float,
    lateral_m: float,
) -> Tuple[float, float, float]:
    local = np.asarray([float(forward_m), float(lateral_m), 0.0, 1.0], dtype=np.float64)
    world = np.asarray(camera_world_matrix, dtype=np.float64) @ local
    return float(world[0]), float(world[1]), float(world[2])


def _ground_height(world: object, x: float, y: float, fallback_z: float) -> float:
    location = base.carla.Location(x=float(x), y=float(y), z=float(fallback_z))
    waypoint = world.get_map().get_waypoint(location, project_to_road=True)
    if waypoint is not None:
        return float(waypoint.transform.location.z)
    return max(0.0, float(fallback_z) - 6.0)


def spawn_controlled_target_in_world_frame(*args: object, **kwargs: object):
    """Fix ego-relative walker placement and optionally realize a close crowd.

    The shared helper is retained for vehicles. Its walker branch expects a
    world camera pose, but the ego platform supplies the camera attachment's
    relative transform. Track B composes that transform here and owns any
    additional crowd actors so the validated shared collector stays unchanged.
    """

    global _CONTROLLED_TARGET_INFO, _OVERLAY_ACTORS
    if str(kwargs.get("kind", "")) != "walker":
        return _SPAWN_CONTROLLED_TARGET(*args, **kwargs)
    world = kwargs["world"]
    anchor_location = kwargs["anchor_location"]
    relative_camera_transform = kwargs["camera_transform"]
    speed_mps = float(kwargs["speed_mps"])
    fwd_dist_m = float(kwargs["fwd_dist_m"])

    anchor_transform = _resolve_anchor_transform(world, anchor_location)
    camera_world_matrix = _compose_camera_world_matrix(
        anchor_transform, relative_camera_transform
    )
    right = np.asarray(camera_world_matrix[:3, 1], dtype=np.float64)
    right[2] = 0.0
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise RuntimeError("controlled-walker camera right vector is degenerate")
    right /= right_norm
    cross = base.carla.Vector3D(x=float(right[0]), y=float(right[1]), z=0.0)
    yaw = math.degrees(math.atan2(float(right[1]), float(right[0])))

    spawned: List[object] = []
    primary = None
    try:
        start_x, start_y, camera_z = _world_location_from_camera_frame(
            camera_world_matrix,
            forward_m=fwd_dist_m,
            lateral_m=_PEDESTRIAN_OVERLAY.target_start_lateral_m,
        )
        ground_z = _ground_height(world, start_x, start_y, camera_z)
        target_blueprint = _fresh_walker_blueprint(
            world, 0, "scenesense_detection_ab_pedestrian_target"
        )
        target_transform = base.carla.Transform(
            base.carla.Location(x=start_x, y=start_y, z=ground_z + 1.0),
            base.carla.Rotation(yaw=yaw),
        )
        primary = world.try_spawn_actor(target_blueprint, target_transform)
        if primary is None:
            raise RuntimeError("controlled walker world-space spawn failed")
        spawned.append(primary)

        candidate_offsets = _pedestrian_crowd_offsets(_PEDESTRIAN_OVERLAY)
        crowd: List[object] = []
        for depth_m, lateral_m in candidate_offsets:
            if len(crowd) >= _PEDESTRIAN_OVERLAY.crowd_count:
                break
            planar_separation = math.hypot(
                depth_m - fwd_dist_m,
                lateral_m - _PEDESTRIAN_OVERLAY.target_start_lateral_m,
            )
            if planar_separation < 1.0:
                continue
            x, y, candidate_camera_z = _world_location_from_camera_frame(
                camera_world_matrix,
                forward_m=depth_m,
                lateral_m=lateral_m,
            )
            candidate_ground_z = _ground_height(world, x, y, candidate_camera_z)
            blueprint = _fresh_walker_blueprint(
                world, len(crowd) + 1, "scenesense_detection_ab_pedestrian_crowd"
            )
            transform = base.carla.Transform(
                base.carla.Location(x=x, y=y, z=candidate_ground_z + 1.0),
                base.carla.Rotation(yaw=yaw),
            )
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                crowd.append(actor)
                spawned.append(actor)

        if len(crowd) < _PEDESTRIAN_OVERLAY.crowd_min_spawned:
            raise RuntimeError(
                "controlled pedestrian crowd realization failed: "
                f"spawned {len(crowd)}, required {_PEDESTRIAN_OVERLAY.crowd_min_spawned}, "
                f"requested {_PEDESTRIAN_OVERLAY.crowd_count}"
            )

        world.tick()
        primary.apply_control(
            base.carla.WalkerControl(direction=cross, speed=float(speed_mps))
        )
        for index, actor in enumerate(crowd):
            direction_scale = -1.0 if index % 2 else 1.0
            actor.apply_control(
                base.carla.WalkerControl(
                    direction=base.carla.Vector3D(
                        x=float(right[0]) * direction_scale,
                        y=float(right[1]) * direction_scale,
                        z=0.0,
                    ),
                    speed=float(_PEDESTRIAN_OVERLAY.crowd_speed_mps),
                )
            )
        _OVERLAY_ACTORS.extend(crowd)
        _CONTROLLED_TARGET_INFO = {
            "actor_id": int(primary.id),
            "type_id": str(getattr(primary, "type_id", "")),
            "role_name": str(primary.attributes.get("role_name", "")),
            "transform": base._carla_transform_payload(primary.get_transform()),
            "commanded_speed_mps": speed_mps,
            "commanded_forward_m": fwd_dist_m,
            "commanded_start_lateral_m": _PEDESTRIAN_OVERLAY.target_start_lateral_m,
            "camera_world_matrix": camera_world_matrix.tolist(),
            "crowd_requested": _PEDESTRIAN_OVERLAY.crowd_count,
            "crowd_spawned": len(crowd),
            "crowd_cleanup_owner": "policy_corpus_overlay",
        }
        print(
            "[controlled-target-overlay] walker "
            f"{primary.type_id} id={primary.id} @ {speed_mps:.1f} m/s; "
            f"world_start=({start_x:.1f},{start_y:.1f},{ground_z:.1f}), "
            f"camera_relative=(forward={fwd_dist_m:.1f},"
            f"lateral={_PEDESTRIAN_OVERLAY.target_start_lateral_m:.1f}); "
            f"close_crowd={len(crowd)}/{_PEDESTRIAN_OVERLAY.crowd_count}"
        )
        return primary, cross
    except Exception:
        # The shared caller does not own the primary until this function returns.
        try:
            base.pole_client._destroy_actors(list(reversed(spawned)))
        except Exception:
            for actor in reversed(spawned):
                try:
                    actor.destroy()
                except Exception:
                    pass
        _OVERLAY_ACTORS = []
        _CONTROLLED_TARGET_INFO = None
        raise


def write_manifest_with_overlay_provenance(
    logger: object, *args: object, **kwargs: object
) -> None:
    """Persist the stripped Track-B controls and exact controlled target ID."""

    _WRITE_MANIFEST(logger, *args, **kwargs)
    manifest_path = Path(logger.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["controlled_target"] = _CONTROLLED_TARGET_INFO
    manifest["policy_corpus_overlay"] = asdict(_PEDESTRIAN_OVERLAY)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config_path = Path(logger.config_path)
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    resolved["policy_corpus_overlay"] = asdict(_PEDESTRIAN_OVERLAY)
    config_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")


def decode_objects_with_diagnostics(
    object_output: "base.torch.Tensor", **kwargs: object
) -> List[Dict[str, float]]:
    """Capture candidate saturation without changing the validated decoder.

    ``pre_topk_above_threshold_count`` is the number of class/heatmap cells at
    or above the live decode threshold before either top-k truncation or NMS.
    The returned detections are still produced by the original decoder.
    """

    tensor = object_output[0] if object_output.ndim == 4 else object_output
    predict_bbox2d = bool(kwargs.get("predict_bbox2d", False))
    heatmap_channels = max(1, int(tensor.shape[0]) - object_reg_channels(predict_bbox2d))
    score_threshold = float(kwargs["score_threshold"])
    with base.torch.inference_mode():
        center = base.torch.sigmoid(tensor[:heatmap_channels])
        pre_topk_count = int((center >= score_threshold).sum().item())
    predictions = _DECODE_OBJECTS(object_output, **kwargs)
    topk = int(kwargs["topk"])
    _DECODE_DIAGNOSTICS.current = {
        "decode_pre_topk_above_threshold_count": pre_topk_count,
        "decode_post_topk_nms_count": int(len(predictions)),
        "decode_topk_limit": topk,
        "decode_topk_saturated": int(pre_topk_count >= topk),
    }
    return predictions


def run_back_half_with_diagnostics(
    worker: "base.FusionRemoteInferenceWorker", payload: Dict[str, object]
) -> Dict[str, object]:
    """Attach same-frame decoder diagnostics to the returned result payload."""

    _DECODE_DIAGNOSTICS.current = None
    result = _RUN_BACK_HALF(worker, payload)
    diagnostics = getattr(_DECODE_DIAGNOSTICS, "current", None)
    if isinstance(diagnostics, dict):
        result.update(diagnostics)
        with _DECODE_BY_FRAME_LOCK:
            _DECODE_BY_FRAME[int(result["frame_id"])] = {
                str(key): int(value) for key, value in diagnostics.items()
            }
    return result


def build_policy_corpus_metrics_row(*args: object, **kwargs: object) -> Dict[str, object]:
    """Expose the capture/render timing already measured by the shared loop."""

    row = _BUILD_FUSION_METRICS_ROW(*args, **kwargs)
    front_stats = kwargs.get("front_stats")
    if not isinstance(front_stats, dict):
        front_stats = {}
    row["camera_frame_wait_ms"] = base._safe_float(
        front_stats.get("camera_frame_wait_ms"), float("nan")
    )
    frame_id = int(kwargs.get("frame_id", -1))
    with _DECODE_BY_FRAME_LOCK:
        diagnostics = _DECODE_BY_FRAME.pop(frame_id, {})
    row["decode_diagnostics_present"] = int(bool(diagnostics))
    row["decode_pre_topk_above_threshold_count"] = base._safe_int(
        diagnostics.get("decode_pre_topk_above_threshold_count"), 0
    )
    row["decode_post_topk_nms_count"] = base._safe_int(
        diagnostics.get("decode_post_topk_nms_count"), 0
    )
    row["decode_topk_limit"] = base._safe_int(diagnostics.get("decode_topk_limit"), 0)
    row["decode_topk_saturated"] = base._safe_int(
        diagnostics.get("decode_topk_saturated"), 0
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
    global _PEDESTRIAN_OVERLAY, _CONTROLLED_TARGET_INFO
    _PEDESTRIAN_OVERLAY, inherited_argv = _parse_overlay_args(sys.argv[1:])
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *inherited_argv]
    _CONTROLLED_TARGET_INFO = None
    base.build_vehicle_ground_truth_rows = build_object_ground_truth_rows
    added_metrics_fields = (
        "camera_frame_wait_ms",
        "decode_diagnostics_present",
        "decode_pre_topk_above_threshold_count",
        "decode_post_topk_nms_count",
        "decode_topk_limit",
        "decode_topk_saturated",
    )
    base.FUSION_METRICS_FIELDS = (
        *base.FUSION_METRICS_FIELDS,
        *(field for field in added_metrics_fields if field not in base.FUSION_METRICS_FIELDS),
    )
    base.build_fusion_metrics_row = build_policy_corpus_metrics_row
    base.decode_objects = decode_objects_with_diagnostics
    base.FusionRemoteInferenceWorker._run_back_half = run_back_half_with_diagnostics
    base._spawn_lead_target = spawn_lead_target_with_synchronized_exact_start
    base._spawn_controlled_target = spawn_controlled_target_in_world_frame
    base.FusionRunLogger.write_manifest = write_manifest_with_overlay_provenance
    # Make the inherited manifest name the actual collection entry point.
    base.__file__ = __file__
    try:
        base.main()
    finally:
        _destroy_overlay_actors()
        sys.argv = original_argv


if __name__ == "__main__":
    main()

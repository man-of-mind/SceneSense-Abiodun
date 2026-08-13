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
import csv
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
_SPAWN_PARKED_EGO = base._spawn_parked_ego_vehicle
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
    ego_ignore_walkers_pct: float = 0.0
    ego_route_control: str = "traffic_manager"
    ego_direct_route_speed_mps: float = 6.0
    ego_direct_yield_to_controlled_pedestrian: bool = False
    ego_direct_yield_hold_s: float = 5.0


_PEDESTRIAN_OVERLAY = ControlledPedestrianOverlay()
_OVERLAY_ACTORS: List[object] = []
_CONTROLLED_TARGET_INFO: Optional[Dict[str, object]] = None
_DIRECT_ROUTE_STATE: Dict[str, object] = {}


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
    parser.add_argument("--ego-ignore-walkers-pct", type=float, default=0.0)
    parser.add_argument(
        "--ego-route-control",
        choices=("traffic_manager", "direct"),
        default="traffic_manager",
    )
    parser.add_argument("--ego-direct-route-speed-mps", type=float, default=6.0)
    parser.add_argument(
        "--ego-direct-yield-to-controlled-pedestrian", action="store_true"
    )
    parser.add_argument("--ego-direct-yield-hold-s", type=float, default=5.0)
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
        ego_ignore_walkers_pct=float(parsed.ego_ignore_walkers_pct),
        ego_route_control=str(parsed.ego_route_control),
        ego_direct_route_speed_mps=float(parsed.ego_direct_route_speed_mps),
        ego_direct_yield_to_controlled_pedestrian=bool(
            parsed.ego_direct_yield_to_controlled_pedestrian
        ),
        ego_direct_yield_hold_s=float(parsed.ego_direct_yield_hold_s),
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
    if not 0.0 <= overlay.ego_ignore_walkers_pct <= 100.0:
        raise ValueError("ego walker-ignore percentage must be within [0, 100]")
    if overlay.ego_direct_route_speed_mps <= 0.0:
        raise ValueError("direct ego route speed must be positive")
    if overlay.ego_direct_yield_hold_s <= 0.0:
        raise ValueError("direct ego pedestrian-yield hold must be positive")
    return overlay, remaining


def _load_direct_route(args: argparse.Namespace) -> List[Tuple[float, float]]:
    path = Path(str(args.ego_fixed_path_progress_csv)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"direct ego route CSV not found: {path}")
    points: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            points.append((float(row["ego_x"]), float(row["ego_y"])))
    if len(points) < 2:
        raise ValueError("direct ego route requires at least two points")
    return points


def _wrap_angle_radians(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _vehicle_in_swept_forward_corridor(
    *,
    forward_m: float,
    lateral_m: float,
    maximum_forward_m: float,
) -> bool:
    """Cover the widening path swept while following a curved route.

    A fixed same-lane lateral bound is unsafe on a bend: a stopped lead can be
    several metres off the ego's instantaneous heading while still lying on
    the route immediately ahead.  Keep the near field selective, then widen
    toward a six-metre cap over the braking look-ahead.
    """

    lateral_limit_m = min(6.0, 2.6 + 0.35 * max(0.0, float(forward_m)))
    return bool(
        0.0 < float(forward_m) <= float(maximum_forward_m)
        and float(lateral_m) <= lateral_limit_m
    )


def _walker_in_ego_forward_corridor(
    *,
    forward_m: float,
    lateral_m: float,
    walker_speed_mps: float,
) -> bool:
    """Protect the direct-route ego from ambient as well as scripted walkers."""

    lateral_limit_m = 3.5 if float(walker_speed_mps) >= 0.2 else 2.2
    return bool(
        0.0 < float(forward_m) <= 15.0
        and float(lateral_m) <= lateral_limit_m
    )


def _apply_direct_ego_route_control(
    actor: object,
    args: argparse.Namespace,
) -> Tuple[int, float, bool]:
    """Follow the UI-authored loop with deterministic per-tick vehicle control."""

    actor_id = int(actor.id)
    if int(_DIRECT_ROUTE_STATE.get("actor_id", -1)) != actor_id:
        points = _load_direct_route(args)
        location = actor.get_location()
        start_index = min(
            range(len(points)),
            key=lambda index: math.hypot(
                points[index][0] - float(location.x),
                points[index][1] - float(location.y),
            ),
        )
        actor.set_autopilot(False, int(args.tm_port))
        _DIRECT_ROUTE_STATE.clear()
        _DIRECT_ROUTE_STATE.update(
            actor_id=actor_id,
            points=points,
            waypoint_index=int(start_index),
        )
        print(
            "Policy-overlay direct ego route controller enabled: "
            f"ego_actor_id={actor_id}, route_points={len(points)}, "
            f"target_speed={_PEDESTRIAN_OVERLAY.ego_direct_route_speed_mps:.1f} m/s"
        )

    points = _DIRECT_ROUTE_STATE["points"]
    index = int(_DIRECT_ROUTE_STATE["waypoint_index"])
    transform = actor.get_transform()
    location = transform.location
    for _unused in range(len(points)):
        target_x, target_y = points[index]
        if math.hypot(target_x - float(location.x), target_y - float(location.y)) >= 4.0:
            break
        index = (index + 1) % len(points)
    # One additional path point is a small look-ahead that damps waypoint
    # oscillation without cutting across the UI route at intersections.
    target_index = (index + 1) % len(points)
    target_x, target_y = points[target_index]
    desired_yaw = math.atan2(
        target_y - float(location.y), target_x - float(location.x)
    )
    heading_error = _wrap_angle_radians(
        desired_yaw - math.radians(float(transform.rotation.yaw))
    )
    velocity = actor.get_velocity()
    speed_mps = math.sqrt(
        float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
    )
    target_speed = float(_PEDESTRIAN_OVERLAY.ego_direct_route_speed_mps)
    turn_scale = max(0.35, 1.0 - abs(heading_error) / math.pi)
    commanded_speed = target_speed * turn_scale
    speed_error = commanded_speed - speed_mps
    throttle = max(0.0, min(0.75, 0.30 * speed_error))
    brake = max(0.0, min(0.65, -0.35 * speed_error))
    steer = max(-0.70, min(0.70, heading_error / math.radians(45.0)))
    control_step = int(_DIRECT_ROUTE_STATE.get("control_step", 0)) + 1
    _DIRECT_ROUTE_STATE["control_step"] = control_step
    # The mixed family contains advisor-managed ambient walkers. They require
    # the same short-horizon collision shield as the controlled crossing; the
    # older role-filtered logic let an ambient walker enter the ego footprint.
    forward = transform.get_forward_vector()
    for walker in actor.get_world().get_actors().filter("walker.pedestrian.*"):
        try:
            walker_location = walker.get_location()
            relative_x = float(walker_location.x) - float(location.x)
            relative_y = float(walker_location.y) - float(location.y)
            forward_m = relative_x * float(forward.x) + relative_y * float(forward.y)
            lateral_m = abs(
                -relative_x * float(forward.y) + relative_y * float(forward.x)
            )
            walker_velocity = walker.get_velocity()
            walker_speed = math.hypot(
                float(walker_velocity.x), float(walker_velocity.y)
            )
        except (AttributeError, RuntimeError):
            continue
        if _walker_in_ego_forward_corridor(
            forward_m=forward_m,
            lateral_m=lateral_m,
            walker_speed_mps=walker_speed,
        ):
            _DIRECT_ROUTE_STATE["yield_until_step"] = max(
                int(_DIRECT_ROUTE_STATE.get("yield_until_step", -1)),
                control_step + 2,
            )
            break
    if _PEDESTRIAN_OVERLAY.ego_direct_yield_to_controlled_pedestrian:
        for walker in actor.get_world().get_actors().filter("walker.pedestrian.*"):
            try:
                role_name = str(walker.attributes.get("role_name", ""))
                if not role_name.startswith("pedestrian_blocker_v4"):
                    continue
                walker_location = walker.get_location()
                relative_x = float(walker_location.x) - float(location.x)
                relative_y = float(walker_location.y) - float(location.y)
                forward_m = relative_x * float(forward.x) + relative_y * float(forward.y)
                lateral_m = abs(
                    -relative_x * float(forward.y) + relative_y * float(forward.x)
                )
                walker_velocity = walker.get_velocity()
                walker_speed = math.hypot(
                    float(walker_velocity.x), float(walker_velocity.y)
                )
            except (AttributeError, RuntimeError):
                continue
            moving_crossing = walker_speed >= 0.2
            cautious_close_approach = forward_m <= 7.0 and speed_mps >= 2.0
            if (
                0.0 < forward_m <= 10.0
                and lateral_m <= 3.0
                and (moving_crossing or cautious_close_approach)
            ):
                hold_frames = int(
                    math.ceil(
                        _PEDESTRIAN_OVERLAY.ego_direct_yield_hold_s
                        * float(base.resolved_world_tick_hz(args))
                    )
                )
                _DIRECT_ROUTE_STATE["yield_until_step"] = control_step + hold_frames
                break
    # Direct route control deliberately bypasses Traffic Manager, so retain a
    # small observable car-following shield rather than rear-ending advisor
    # traffic that shares the UI route. This is evaluated every synchronous
    # control tick and refreshed while a lead remains inside the route's swept
    # forward corridor, including leads around a bend.
    forward = transform.get_forward_vector()
    for vehicle in actor.get_world().get_actors().filter("vehicle.*"):
        if int(vehicle.id) == actor_id:
            continue
        try:
            other_location = vehicle.get_location()
            relative_x = float(other_location.x) - float(location.x)
            relative_y = float(other_location.y) - float(location.y)
            forward_m = relative_x * float(forward.x) + relative_y * float(forward.y)
            lateral_m = abs(
                -relative_x * float(forward.y) + relative_y * float(forward.x)
            )
            other_velocity = vehicle.get_velocity()
            other_speed_mps = math.hypot(
                float(other_velocity.x), float(other_velocity.y)
            )
        except (AttributeError, RuntimeError):
            continue
        stopping_margin_m = max(
            10.0,
            7.0 + max(0.0, speed_mps ** 2 - other_speed_mps ** 2) / 7.0,
        )
        if _vehicle_in_swept_forward_corridor(
            forward_m=forward_m,
            lateral_m=lateral_m,
            maximum_forward_m=stopping_margin_m,
        ):
            _DIRECT_ROUTE_STATE["yield_until_step"] = max(
                int(_DIRECT_ROUTE_STATE.get("yield_until_step", -1)),
                control_step + 2,
            )
            break
    yielding = control_step <= int(_DIRECT_ROUTE_STATE.get("yield_until_step", -1))
    if yielding:
        throttle = 0.0
        brake = 1.0
    actor.apply_control(
        base.carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
        )
    )
    _DIRECT_ROUTE_STATE["waypoint_index"] = int(index)
    _DIRECT_ROUTE_STATE["heading_error"] = float(heading_error)
    _DIRECT_ROUTE_STATE["yielding"] = bool(yielding)
    return int(index), float(heading_error), bool(yielding)


def on_policy_control_tick(
    *,
    world: object,
    anchor_actor: object,
    args: argparse.Namespace,
    frame_id: int,
) -> None:
    """Run the UI-route ego controller on the 20 Hz control clock."""

    del world
    if _PEDESTRIAN_OVERLAY.ego_route_control != "direct":
        return
    if anchor_actor is None:
        raise RuntimeError("direct ego route controller requires an ego actor")
    _apply_direct_ego_route_control(anchor_actor, args)
    _DIRECT_ROUTE_STATE["last_control_frame_id"] = int(frame_id)


def spawn_parked_ego_with_tm_overrides(
    *,
    world: object,
    args: argparse.Namespace,
) -> object:
    """Apply the explicitly configured pedestrian-smoke TM exception.

    The advisor crossing trigger requires the ego to approach a stationary
    walker.  Traffic Manager otherwise brakes before the trigger distance is
    reached.  Keep this exception in the derived collector, disabled by
    default, and destroy the actor if the configured override cannot be
    applied so an invalid run cannot continue silently.
    """

    actor = _SPAWN_PARKED_EGO(world=world, args=args)
    percentage = float(_PEDESTRIAN_OVERLAY.ego_ignore_walkers_pct)
    if percentage <= 0.0:
        return actor
    try:
        client = base.carla.Client(str(args.host), int(args.port))
        client.set_timeout(10.0)
        traffic_manager = client.get_trafficmanager(int(args.tm_port))
        traffic_manager.ignore_walkers_percentage(actor, percentage)
    except Exception as exc:
        try:
            actor.destroy()
        except Exception:
            pass
        raise RuntimeError(
            "configured ego walker-ignore override could not be applied: "
            f"percentage={percentage:.1f}, tm_port={int(args.tm_port)}"
        ) from exc
    print(
        "Policy-overlay Traffic Manager override applied: "
        f"ego_actor_id={int(actor.id)}, ignore_walkers={percentage:.1f}%, "
        f"tm_port={int(args.tm_port)}"
    )
    return actor


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
    manifest["clock_contract"] = {
        "world_control_hz": float(base.resolved_world_tick_hz(logger.args)),
        "sensor_detection_hz": float(logger.args.fps),
        "control_ticks_per_sensor_frame": int(
            base.synchronous_ticks_per_sensor_frame(logger.args)
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config_path = Path(logger.config_path)
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    resolved["policy_corpus_overlay"] = asdict(_PEDESTRIAN_OVERLAY)
    resolved["clock_contract"] = manifest["clock_contract"]
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
    row["ego_route_control_mode"] = _PEDESTRIAN_OVERLAY.ego_route_control
    row["ego_route_waypoint_index"] = ""
    row["ego_route_heading_error_deg"] = ""
    row["ego_route_yielding"] = ""
    row["world_control_tick_hz"] = float(base.resolved_world_tick_hz(kwargs["args"]))
    row["sensor_detection_hz"] = float(kwargs["args"].fps)
    row["control_ticks_per_sensor_frame"] = base._safe_int(
        front_stats.get("control_ticks_per_sensor_frame"), 0
    )
    if _PEDESTRIAN_OVERLAY.ego_route_control == "direct":
        row["ego_route_waypoint_index"] = int(
            _DIRECT_ROUTE_STATE.get("waypoint_index", -1)
        )
        row["ego_route_heading_error_deg"] = math.degrees(
            float(_DIRECT_ROUTE_STATE.get("heading_error", float("nan")))
        )
        row["ego_route_yielding"] = int(
            bool(_DIRECT_ROUTE_STATE.get("yielding", False))
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
    _DIRECT_ROUTE_STATE.clear()
    base.build_vehicle_ground_truth_rows = build_object_ground_truth_rows
    added_metrics_fields = (
        "camera_frame_wait_ms",
        "decode_diagnostics_present",
        "decode_pre_topk_above_threshold_count",
        "decode_post_topk_nms_count",
        "decode_topk_limit",
        "decode_topk_saturated",
        "ego_route_control_mode",
        "ego_route_waypoint_index",
        "ego_route_heading_error_deg",
        "ego_route_yielding",
        "world_control_tick_hz",
        "sensor_detection_hz",
        "control_ticks_per_sensor_frame",
    )
    base.FUSION_METRICS_FIELDS = (
        *base.FUSION_METRICS_FIELDS,
        *(field for field in added_metrics_fields if field not in base.FUSION_METRICS_FIELDS),
    )
    base.build_fusion_metrics_row = build_policy_corpus_metrics_row
    base.decode_objects = decode_objects_with_diagnostics
    base.FusionRemoteInferenceWorker._run_back_half = run_back_half_with_diagnostics
    base._spawn_parked_ego_vehicle = spawn_parked_ego_with_tm_overrides
    base._spawn_lead_target = spawn_lead_target_with_synchronized_exact_start
    base._spawn_controlled_target = spawn_controlled_target_in_world_frame
    base.on_synchronous_world_tick = on_policy_control_tick
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

#!/usr/bin/env python3
"""Run the single six-case controlled renderer z-buffer qualification."""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .actor_state import (
    capture_actor_state,
    capture_walker_bones,
    carla_transform_from_matrix,
    configure_clone,
    set_blueprint_attributes,
    walker_bone_pose_error,
)
from .core import (
    TAU_EMPTY_M,
    TAU_MATCH_M,
    TRANSFORM_TOLERANCE,
    WALKER_BONE_TOLERANCE,
    ZBufferVisibilityError,
    compute_zbuffer_visibility,
    decode_depth_bgra,
    image_bgra,
    mask_iou,
    reproduce_transform_matrix,
    sha256,
    transform_matrix,
    transform_payload,
    write_json_x,
    write_npy_x,
    write_png_x,
)


WIDTH, HEIGHT, FOV = 1280, 720, 120.0
WORLD_DELTA_S = 0.05
REFERENCE_CAMERA_Z_M = 800.0
EMPTY_NEARBY_DEPTH_M = 900.0
PREFERRED_OCCLUDER_DEPTH_M = 4.5
OCCLUDER_CAMERA_MARGIN_M = 0.50
OCCLUDER_TARGET_MARGIN_M = 0.50
OCCLUDER_BLUEPRINTS = (
    "static.prop.box03",
    "static.prop.box02",
    "static.prop.box01",
    "static.prop.streetbarrier",
    "static.prop.container",
)
VEHICLE_SEMANTIC_TAG = 14
QUALIFIED = "PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_QUALIFIED_AWAITING_TRAFFIC_SMOKE"
BLOCKED = "PUBLICATION_ZBUFFER_VISIBILITY_CONTROLLED_BLOCKED"
IMPLEMENTATION_FAILED = "PUBLICATION_ZBUFFER_VISIBILITY_IMPLEMENTATION_FAILED"


class QualificationBlocked(ZBufferVisibilityError):
    """A scientific qualification gate failed without a permitted fallback."""


def _camera_blueprint(world: Any, type_id: str) -> Any:
    blueprint = world.get_blueprint_library().find(type_id)
    for key, value in (
        ("image_size_x", WIDTH),
        ("image_size_y", HEIGHT),
        ("fov", FOV),
        ("sensor_tick", 0.0),
    ):
        blueprint.set_attribute(key, str(value))
    return blueprint


def _intrinsics() -> np.ndarray:
    focal = WIDTH / (2.0 * math.tan(math.radians(FOV) / 2.0))
    return np.asarray(
        [[focal, 0.0, WIDTH / 2.0], [0.0, focal, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _wait_exact(sensor_queue: queue.Queue, frame: int, label: str) -> Any:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            item = sensor_queue.get(timeout=max(0.01, deadline - time.monotonic()))
        except queue.Empty:
            break
        if int(item.frame) < int(frame):
            continue
        if int(item.frame) == int(frame):
            return item
        raise ZBufferVisibilityError(
            f"{label} overshot frame {frame} with frame {int(item.frame)}"
        )
    raise ZBufferVisibilityError(f"{label} missed frame {frame}")


def _drain(*queues: queue.Queue) -> None:
    for sensor_queue in queues:
        while True:
            try:
                sensor_queue.get_nowait()
            except queue.Empty:
                break


def _choose(world: Any, preferred: str, pattern: str) -> Any:
    library = world.get_blueprint_library()
    try:
        return library.find(preferred)
    except RuntimeError:
        candidates = sorted(library.filter(pattern), key=lambda value: value.id)
        if not candidates:
            raise ZBufferVisibilityError(f"no blueprint matches {preferred}/{pattern}")
        return candidates[0]


def _spawn_disabled(world: Any, blueprint: Any, transform: Any) -> Any:
    import carla

    staging = carla.Transform(carla.Location(x=0.0, y=0.0, z=950.0), transform.rotation)
    actor = world.try_spawn_actor(blueprint, staging)
    if actor is None:
        raise ZBufferVisibilityError(f"actor spawn failed: {blueprint.id}")
    try:
        actor.set_simulate_physics(False)
    except (AttributeError, RuntimeError):
        pass
    actor.set_transform(transform)
    return actor


def _rgb_bgr(image: Any) -> np.ndarray:
    return image_bgra(image)[:, :, :3]


def _persist_depth(directory: Path, stem: str, raw: np.ndarray, depth_m: np.ndarray) -> dict[str, Any]:
    raw_path = directory / f"{stem}_raw_bgra.png"
    depth_path = directory / f"{stem}_metres_f64.npy"
    return {
        "raw_bgra_path": str(raw_path),
        "raw_bgra_sha256": write_png_x(raw_path, raw),
        "metres_f64_path": str(depth_path),
        "metres_f64_sha256": write_npy_x(depth_path, depth_m),
        "minimum_m": float(np.min(depth_m)),
        "maximum_m": float(np.max(depth_m)),
        "finite": bool(np.all(np.isfinite(depth_m))),
    }


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise QualificationBlocked("actor-only depth support is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_world_points(actor: Any) -> np.ndarray:
    bbox = actor.bounding_box
    extent = bbox.extent
    center = np.asarray([bbox.location.x, bbox.location.y, bbox.location.z])
    offsets = np.asarray(
        [
            [sx * extent.x, sy * extent.y, sz * extent.z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    local = center[None, :] + offsets
    homogeneous = np.concatenate([local, np.ones((8, 1))], axis=1)
    return (transform_matrix(actor.get_transform()) @ homogeneous.T).T[:, :3]


def safe_occluder_center_depth(
    camera_forward_corner_offsets_m: np.ndarray,
    *,
    preferred_depth_m: float = PREFERRED_OCCLUDER_DEPTH_M,
    camera_margin_m: float = OCCLUDER_CAMERA_MARGIN_M,
) -> float:
    """Smallest centre depth preserving the preferred depth and camera margin."""
    offsets = np.asarray(camera_forward_corner_offsets_m, dtype=np.float64)
    if offsets.shape != (8,) or not np.all(np.isfinite(offsets)):
        raise ZBufferVisibilityError("occluder camera-forward offsets must be eight finite values")
    if preferred_depth_m <= 0.0 or camera_margin_m <= 0.0:
        raise ZBufferVisibilityError("occluder depth and camera margin must be positive")
    centre = max(float(preferred_depth_m), float(camera_margin_m) - float(np.min(offsets)))
    if float(np.min(offsets + centre)) < float(camera_margin_m) - 1e-12:
        raise ZBufferVisibilityError("safe occluder centre-depth calculation violated margin")
    return centre


def _bbox_camera_points(actor: Any, camera_transform: Any) -> np.ndarray:
    inverse = np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64)
    world = _bbox_world_points(actor)
    homogeneous = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1)
    return (inverse @ homogeneous.T).T[:, :3]


def _project_bbox(actor: Any, camera_transform: Any, intrinsics: np.ndarray) -> tuple[float, ...]:
    camera = _bbox_camera_points(actor, camera_transform)
    if np.any(camera[:, 0] <= 0.0):
        raise ZBufferVisibilityError("occluder bounding box crosses the camera plane")
    u = intrinsics[0, 2] + camera[:, 1] / camera[:, 0] * intrinsics[0, 0]
    v = intrinsics[1, 2] - camera[:, 2] / camera[:, 0] * intrinsics[1, 1]
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def _intended_occluder_rotation(camera_transform: Any) -> Any:
    import carla

    return carla.Rotation(
        pitch=float(camera_transform.rotation.pitch),
        yaw=float(camera_transform.rotation.yaw) + 90.0,
        roll=float(camera_transform.rotation.roll),
    )


def _camera_forward_corner_offsets(
    occluder: Any, camera_transform: Any, rotation: Any
) -> np.ndarray:
    import carla

    bbox = occluder.bounding_box
    extent = bbox.extent
    local_offsets = np.asarray(
        [
            [sx * extent.x, sy * extent.y, sz * extent.z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    camera_rotation_inverse = np.asarray(
        camera_transform.get_inverse_matrix(), dtype=np.float64
    )[:3, :3]
    actor_rotation = transform_matrix(carla.Transform(carla.Location(), rotation))[:3, :3]
    return (camera_rotation_inverse @ actor_rotation @ local_offsets.T).T[:, 0]


def _set_occluder_bbox_center(
    occluder: Any,
    camera_transform: Any,
    intrinsics: np.ndarray,
    *,
    target_u: float,
    target_v: float,
    center_depth_m: float,
) -> None:
    import carla

    rotation = _intended_occluder_rotation(camera_transform)
    rotation_matrix = transform_matrix(carla.Transform(carla.Location(), rotation))[:3, :3]
    local_center = np.asarray(
        [
            occluder.bounding_box.location.x,
            occluder.bounding_box.location.y,
            occluder.bounding_box.location.z,
        ],
        dtype=np.float64,
    )
    point_camera = np.asarray(
        [
            center_depth_m,
            (target_u - intrinsics[0, 2]) * center_depth_m / intrinsics[0, 0],
            -(target_v - intrinsics[1, 2]) * center_depth_m / intrinsics[1, 1],
            1.0,
        ]
    )
    desired_center = (transform_matrix(camera_transform) @ point_camera)[:3]
    origin = desired_center - rotation_matrix @ local_center
    occluder.set_transform(
        carla.Transform(
            carla.Location(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
            rotation,
        )
    )


def _occluder_geometry(
    occluder: Any, camera_transform: Any, intrinsics: np.ndarray
) -> dict[str, Any]:
    points = _bbox_camera_points(occluder, camera_transform)
    bbox_center_world = transform_matrix(occluder.get_transform()) @ np.asarray(
        [
            occluder.bounding_box.location.x,
            occluder.bounding_box.location.y,
            occluder.bounding_box.location.z,
            1.0,
        ]
    )
    bbox_center_camera = np.asarray(
        camera_transform.get_inverse_matrix(), dtype=np.float64
    ) @ bbox_center_world
    return {
        "camera_forward_corner_depths_m": [float(value) for value in points[:, 0]],
        "camera_forward_minimum_m": float(np.min(points[:, 0])),
        "camera_forward_center_m": float(bbox_center_camera[0]),
        "camera_forward_maximum_m": float(np.max(points[:, 0])),
        "projected_bbox_xyxy": list(_project_bbox(occluder, camera_transform, intrinsics)),
    }


def _covers_support(projected: tuple[float, ...] | list[float], support: np.ndarray) -> bool:
    x0, y0, x1, y1 = _mask_bbox(support)
    return bool(
        float(projected[0]) <= x0
        and float(projected[1]) <= y0
        and float(projected[2]) >= x1
        and float(projected[3]) >= y1
    )


def _park_occluder(occluder: Any, camera_transform: Any) -> None:
    import carla

    matrix = transform_matrix(camera_transform)
    parked = matrix @ np.asarray([-50.0, 0.0, 0.0, 1.0])
    occluder.set_transform(
        carla.Transform(
            carla.Location(x=float(parked[0]), y=float(parked[1]), z=float(parked[2])),
            camera_transform.rotation,
        )
    )


def _select_safe_occluder(
    world: Any,
    camera_transform: Any,
    intrinsics: np.ndarray,
    references: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    import carla

    library = world.get_blueprint_library()
    for type_id in OCCLUDER_BLUEPRINTS:
        try:
            blueprint = library.find(type_id)
        except RuntimeError:
            continue
        try:
            candidate = _spawn_disabled(
                world,
                blueprint,
                carla.Transform(
                    carla.Location(x=-50.0, y=0.0, z=float(camera_transform.location.z)),
                    carla.Rotation(),
                ),
            )
        except ZBufferVisibilityError:
            continue
        profiles: dict[str, dict[str, Any]] = {}
        accepted = True
        try:
            rotation = _intended_occluder_rotation(camera_transform)
            offsets = _camera_forward_corner_offsets(candidate, camera_transform, rotation)
            center_depth = safe_occluder_center_depth(offsets)
            for class_name in ("vehicle", "person"):
                reference = references[class_name]
                nearest_target_depth = float(
                    np.min(reference["actor_depth"][reference["support"]])
                )
                geometry = _place_occluder(
                    candidate,
                    camera_transform,
                    intrinsics,
                    reference["support"],
                    "full",
                    center_depth,
                    nearest_target_depth,
                )
                profiles[class_name] = {
                    "center_depth_m": center_depth,
                    "nearest_target_support_surface_m": nearest_target_depth,
                    "camera_forward_minimum_m": geometry["camera_forward_minimum_m"],
                    "camera_forward_center_m": geometry["camera_forward_center_m"],
                    "camera_forward_maximum_m": geometry["camera_forward_maximum_m"],
                    "projected_bbox_xyxy": geometry["projected_bbox_xyxy"],
                }
            if accepted:
                _park_occluder(candidate, camera_transform)
                return candidate, profiles
        except ZBufferVisibilityError:
            accepted = False
        finally:
            if not accepted:
                try:
                    candidate.destroy()
                    world.tick()
                except RuntimeError:
                    pass
    raise ZBufferVisibilityError(
        "no fixed-order opaque prop satisfies camera, target-depth and full-coverage geometry"
    )


def _place_occluder(
    occluder: Any,
    camera_transform: Any,
    intrinsics: np.ndarray,
    support: np.ndarray,
    condition: str,
    center_depth_m: float,
    nearest_target_depth_m: float,
) -> dict[str, Any]:
    x0, y0, x1, y1 = _mask_bbox(support)
    target_u, target_v = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
    _set_occluder_bbox_center(
        occluder,
        camera_transform,
        intrinsics,
        target_u=target_u,
        target_v=target_v,
        center_depth_m=center_depth_m,
    )

    # Pure geometry alignment: center the projected opaque prop for full, or put
    # its left projected bound through the actor-support centre for partial.
    for _ in range(8):
        bounds = _project_bbox(occluder, camera_transform, intrinsics)
        desired_u = target_u if condition == "full" else target_u + (bounds[2] - bounds[0]) / 2.0
        current_u = (bounds[0] + bounds[2]) / 2.0
        current_v = (bounds[1] + bounds[3]) / 2.0
        du, dv = desired_u - current_u, target_v - current_v
        if abs(du) <= 0.05 and abs(dv) <= 0.05:
            break
        current = occluder.get_transform()
        translation_camera = np.asarray(
            [
                0.0,
                du * center_depth_m / intrinsics[0, 0],
                -dv * center_depth_m / intrinsics[1, 1],
                0.0,
            ]
        )
        translation_world = transform_matrix(camera_transform) @ translation_camera
        current.location.x += float(translation_world[0])
        current.location.y += float(translation_world[1])
        current.location.z += float(translation_world[2])
        occluder.set_transform(current)
    geometry = _occluder_geometry(occluder, camera_transform, intrinsics)
    bounds = geometry["projected_bbox_xyxy"]
    if geometry["camera_forward_minimum_m"] < OCCLUDER_CAMERA_MARGIN_M - 1e-6:
        raise ZBufferVisibilityError("final occluder violates camera-plane margin")
    if (
        geometry["camera_forward_maximum_m"]
        > nearest_target_depth_m - OCCLUDER_TARGET_MARGIN_M + 1e-6
    ):
        raise ZBufferVisibilityError("final occluder violates target-depth margin")
    vertical_center = (bounds[1] + bounds[3]) / 2.0
    if abs(vertical_center - target_v) > 0.5:
        raise ZBufferVisibilityError("final occluder is not vertically centred")
    if float(bounds[1]) > y0 or float(bounds[3]) < y1:
        raise ZBufferVisibilityError("final occluder does not vertically cover target support")
    if condition == "full":
        if abs((bounds[0] + bounds[2]) / 2.0 - target_u) > 0.5:
            raise ZBufferVisibilityError("final full occluder is not horizontally centred")
        if not _covers_support(bounds, support):
            raise ZBufferVisibilityError("final full occluder does not cover target support box")
    if condition == "partial" and abs(bounds[0] - target_u) > 0.5:
        raise ZBufferVisibilityError("final partial occluder edge misses support centre")
    return {
        "blueprint": str(occluder.type_id),
        "condition": condition,
        "transform": transform_payload(occluder.get_transform()),
        "nearest_target_support_surface_m": nearest_target_depth_m,
        "camera_margin_m": OCCLUDER_CAMERA_MARGIN_M,
        "target_depth_margin_m": OCCLUDER_TARGET_MARGIN_M,
        **geometry,
    }


def optional_vehicle_instance_diagnostic(
    raw_bgra: np.ndarray, depth_support: np.ndarray
) -> dict[str, Any]:
    """Return the amendment-001 vehicle diagnostic without blocking depth gates."""
    semantic = raw_bgra[:, :, 2]
    rendered = raw_bgra[:, :, 0].astype(np.uint32)
    rendered += raw_bgra[:, :, 1].astype(np.uint32) << np.uint32(8)
    vehicle_pixels = semantic == VEHICLE_SEMANTIC_TAG
    tokens = sorted(int(value) for value in np.unique(rendered[vehicle_pixels]) if int(value) > 0)
    if len(tokens) != 1:
        return {
            "instance_diagnostic_available": False,
            "instance_diagnostic_unavailable_reason": (
                f"isolated vehicle rendered-token discovery is unavailable or ambiguous: {tokens}"
            ),
            "vehicle_instance_rendered_token": None,
            "vehicle_depth_support_vs_instance_iou": None,
            "instance_component_mask": None,
        }
    token = tokens[0]
    component = vehicle_pixels & (rendered == token)
    return {
        "instance_diagnostic_available": True,
        "instance_diagnostic_unavailable_reason": None,
        "vehicle_instance_rendered_token": token,
        "vehicle_depth_support_vs_instance_iou": mask_iou(depth_support, component),
        "instance_component_mask": component,
    }


def _depth_difference_visualization(difference: np.ndarray, support: np.ndarray) -> np.ndarray:
    import cv2

    scale = np.minimum(difference / 0.20, 1.0)
    gray = np.rint(scale * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~support] = 0
    return color


def _crop_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _mask_bbox(mask)
    width, height = x1 - x0, y1 - y0
    side = max(120, int(math.ceil(max(width, height) * 2.4)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left, top = max(0, cx - side // 2), max(0, cy - side // 2)
    right, bottom = min(WIDTH, left + side), min(HEIGHT, top + side)
    left, top = max(0, right - side), max(0, bottom - side)
    return left, top, right, bottom


def _contact_sheet(
    cases: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> str:
    import cv2

    tile_size, text_height = 220, 78
    panel_w, panel_h = 4 * tile_size, tile_size + text_height
    sheet = np.full((2 * panel_h, 3 * panel_w, 3), 245, dtype=np.uint8)
    for row_index, class_name in enumerate(("vehicle", "person")):
        reference = references[class_name]
        bounds = _crop_bounds(reference["support"])
        x0, y0, x1, y1 = bounds
        for column_index, condition in enumerate(("clear", "partial", "full")):
            case = cases[f"{class_name}_{condition}"]
            support = reference["support"]
            visible = case["visible"]
            a_view = np.where(support[:, :, None], (255, 255, 255), 0).astype(np.uint8)
            v_view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            v_view[visible] = (0, 255, 0)
            images = (
                case["rgb"],
                a_view,
                v_view,
                _depth_difference_visualization(case["difference"], support),
            )
            origin_x, origin_y = column_index * panel_w, row_index * panel_h
            for tile_index, (label, image) in enumerate(
                zip(("RGB", "A_i", "V_i", "|Dscene-Dactor|"), images)
            ):
                interpolation = cv2.INTER_LINEAR if tile_index in (0, 3) else cv2.INTER_NEAREST
                tile = cv2.resize(
                    image[y0:y1, x0:x1], (tile_size, tile_size), interpolation=interpolation
                )
                tx = origin_x + tile_index * tile_size
                sheet[origin_y : origin_y + tile_size, tx : tx + tile_size] = tile
                cv2.putText(
                    sheet,
                    label,
                    (tx + 5, origin_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0) if tile_index == 1 else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            metrics = case["metrics"]
            pose = reference["transform_error"]
            bone = reference["walker_bone_error"]
            lines = (
                f"{class_name} {condition}  range={case['range_m']:.3f}m  "
                f"visibility={metrics['visibility']:.6f}  A={metrics['support_pixels']} "
                f"V={metrics['visible_pixels']}",
                f"actor/reference pose err={pose:.3g}  walker pose err="
                f"{'n/a' if bone is None else f'{bone:.3g}'}  tau_empty=tau_match=0.02m",
            )
            for line_index, line in enumerate(lines):
                cv2.putText(
                    sheet,
                    line,
                    (origin_x + 7, origin_y + tile_size + 27 + line_index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
    return write_png_x(path, sheet)


def _public_case(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"rgb", "scene_depth", "visible", "difference"}
    }


def _emit_terminal(output_dir: Path, terminal: str, failure: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "controlled_qualification_failure.json"
    if not evidence_path.exists():
        write_json_x(
            evidence_path,
            {
                "schema": "publication_zbuffer_visibility_controlled_failure_v2",
                "terminal": terminal,
                "failure": failure,
            },
        )
    marker = output_dir / terminal
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(terminal + "\n")


def run(host: str, port: int, output_dir: Path) -> dict[str, Any]:
    import carla

    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.publication_zbuffer_visibility_evaluation_v2.protocol import (
        load_registered_protocol,
    )

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    registration = load_registered_protocol()
    client = carla.Client(host, int(port))
    client.set_timeout(120.0)
    world = client.load_world("Town10HD_Opt", True)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = WORLD_DELTA_S
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    actors: list[Any] = []
    queues = {name: queue.Queue() for name in ("rgb", "scene_depth", "reference_depth", "reference_instance")}
    references: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    cleanup: dict[str, Any] = {}
    try:
        normal_camera_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=500.0), carla.Rotation()
        )
        reference_camera_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=REFERENCE_CAMERA_Z_M), carla.Rotation()
        )
        for name, type_id, transform in (
            ("rgb", "sensor.camera.rgb", normal_camera_transform),
            ("scene_depth", "sensor.camera.depth", normal_camera_transform),
            ("reference_depth", "sensor.camera.depth", reference_camera_transform),
            (
                "reference_instance",
                "sensor.camera.instance_segmentation",
                reference_camera_transform,
            ),
        ):
            sensor = world.spawn_actor(_camera_blueprint(world, type_id), transform)
            sensor.listen(queues[name].put)
            actors.append(sensor)

        vehicle_bp = _choose(world, "vehicle.audi.a2", "vehicle.*")
        walker_bp = _choose(world, "walker.pedestrian.0001", "walker.pedestrian.*")
        vehicle = _spawn_disabled(
            world,
            vehicle_bp,
            carla.Transform(
                carla.Location(x=16.0, y=-4.0, z=498.5), carla.Rotation()
            ),
        )
        walker = _spawn_disabled(
            world,
            walker_bp,
            carla.Transform(
                carla.Location(x=12.0, y=3.0, z=498.3), carla.Rotation()
            ),
        )
        actors.extend((vehicle, walker))
        for _ in range(3):
            world.tick()
        _drain(*queues.values())

        intrinsics = _intrinsics()
        states = {
            "vehicle": capture_actor_state(vehicle, normal_camera_transform, "vehicle"),
            "person": capture_actor_state(walker, normal_camera_transform, "person"),
        }

        # One empty reference frame. Normal-scene actors are far outside this
        # camera's frustum and no reference clone exists yet.
        empty_frame = int(world.tick())
        empty_image = _wait_exact(queues["reference_depth"], empty_frame, "empty reference depth")
        _wait_exact(queues["reference_instance"], empty_frame, "empty reference instance")
        empty_raw = image_bgra(empty_image)
        empty_depth = decode_depth_bgra(empty_raw)
        empty_nearby_pixels = int(np.count_nonzero(empty_depth < EMPTY_NEARBY_DEPTH_M))
        empty_artifacts = _persist_depth(output_dir / "reference", "empty_depth", empty_raw, empty_depth)
        artifact_hashes.update(
            {
                "reference/empty_depth_raw_bgra.png": empty_artifacts["raw_bgra_sha256"],
                "reference/empty_depth_metres_f64.npy": empty_artifacts["metres_f64_sha256"],
            }
        )

        for class_name in ("vehicle", "person"):
            state = states[class_name]
            blueprint = world.get_blueprint_library().find(str(state["blueprint"]))
            set_blueprint_attributes(blueprint, state["blueprint_attributes"])
            desired = reproduce_transform_matrix(
                reference_camera_transform,
                np.asarray(state["camera_relative_actor_matrix"], dtype=np.float64),
            )
            clone = world.try_spawn_actor(blueprint, carla_transform_from_matrix(desired))
            if clone is None:
                raise QualificationBlocked(f"cannot spawn isolated {class_name} clone")
            try:
                configure_clone(clone, state)
                clone.set_transform(carla_transform_from_matrix(desired))
                for _ in range(2):
                    world.tick()
                clone.set_transform(carla_transform_from_matrix(desired))
                reference_frame = int(world.tick())
                depth_image = _wait_exact(
                    queues["reference_depth"], reference_frame, f"{class_name} actor depth"
                )
                instance_image = _wait_exact(
                    queues["reference_instance"], reference_frame, f"{class_name} reference diagnostic"
                )
                actual = transform_matrix(clone.get_transform())
                transform_error = float(np.max(np.abs(actual - desired)))
                bone_error = None
                if class_name == "person":
                    bone_error = walker_bone_pose_error(
                        list(state["walker_bones"]), capture_walker_bones(clone)
                    )
                actor_raw = image_bgra(depth_image)
                actor_depth = decode_depth_bgra(actor_raw)
                try:
                    support_result = compute_zbuffer_visibility(
                        empty_depth, actor_depth, actor_depth
                    )
                except ZBufferVisibilityError as exc:
                    raise QualificationBlocked(f"{class_name} actor-only depth: {exc}") from exc
                support = support_result["support"]
                reference_dir = output_dir / "reference" / class_name
                depth_artifacts = _persist_depth(
                    reference_dir, "actor_only_depth", actor_raw, actor_depth
                )
                support_path = reference_dir / "A_i.png"
                support_hash = write_png_x(
                    support_path, np.where(support, 255, 0).astype(np.uint8)
                )
                instance_diagnostic = {
                    "instance_diagnostic_available": None,
                    "instance_diagnostic_unavailable_reason": None,
                    "vehicle_instance_rendered_token": None,
                    "vehicle_depth_support_vs_instance_iou": None,
                    "instance_component_mask": None,
                }
                if class_name == "vehicle":
                    instance_raw = image_bgra(instance_image)
                    instance_raw_path = reference_dir / "vehicle_instance_raw_bgra.png"
                    artifact_hashes[str(instance_raw_path.relative_to(output_dir))] = write_png_x(
                        instance_raw_path, instance_raw
                    )
                    instance_diagnostic = optional_vehicle_instance_diagnostic(
                        instance_raw, support
                    )
                    component = instance_diagnostic["instance_component_mask"]
                    if component is not None:
                        instance_mask_path = reference_dir / "vehicle_instance_component.png"
                        artifact_hashes[
                            str(instance_mask_path.relative_to(output_dir))
                        ] = write_png_x(
                            instance_mask_path,
                            np.where(component, 255, 0).astype(np.uint8),
                        )
                references[class_name] = {
                    "actor_id": int(state["actor_id"]),
                    "class_name": class_name,
                    "source_actor_state": state,
                    "source_frame": None,
                    "reference_frame": int(reference_frame),
                    "reference_timestamp": float(depth_image.timestamp),
                    "reference_clone_actor_id": int(clone.id),
                    "support": support,
                    "actor_depth": actor_depth,
                    "support_pixels": int(support_result["support_pixels"]),
                    "transform_error": transform_error,
                    "walker_bone_error": bone_error,
                    "instance_diagnostic_available": instance_diagnostic[
                        "instance_diagnostic_available"
                    ],
                    "instance_diagnostic_unavailable_reason": instance_diagnostic[
                        "instance_diagnostic_unavailable_reason"
                    ],
                    "vehicle_instance_rendered_token": instance_diagnostic[
                        "vehicle_instance_rendered_token"
                    ],
                    "vehicle_depth_support_vs_instance_iou": instance_diagnostic[
                        "vehicle_depth_support_vs_instance_iou"
                    ],
                    "actor_depth_artifacts": depth_artifacts,
                    "support_path": str(support_path),
                    "support_sha256": support_hash,
                }
                artifact_hashes[str(support_path.relative_to(output_dir))] = support_hash
                artifact_hashes[
                    str((reference_dir / "actor_only_depth_raw_bgra.png").relative_to(output_dir))
                ] = depth_artifacts["raw_bgra_sha256"]
                artifact_hashes[
                    str((reference_dir / "actor_only_depth_metres_f64.npy").relative_to(output_dir))
                ] = depth_artifacts["metres_f64_sha256"]
                reference_record_path = reference_dir / "reference_record.json"
                reference_record = {
                    key: value
                    for key, value in references[class_name].items()
                    if key not in {"support", "actor_depth"}
                }
                artifact_hashes[
                    str(reference_record_path.relative_to(output_dir))
                ] = write_json_x(reference_record_path, reference_record)
            finally:
                clone.destroy()
                world.tick()

        occluder, occluder_profiles = _select_safe_occluder(
            world, normal_camera_transform, intrinsics, references
        )
        actors.append(occluder)
        targets = {"vehicle": vehicle, "person": walker}
        for class_name in ("vehicle", "person"):
            reference = references[class_name]
            for condition in ("clear", "partial", "full"):
                if condition == "clear":
                    _park_occluder(occluder, normal_camera_transform)
                    occluder_record = {
                        "blueprint": str(occluder.type_id),
                        "condition": "clear",
                        "parked_behind_camera": True,
                    }
                else:
                    occluder_record = _place_occluder(
                        occluder,
                        normal_camera_transform,
                        intrinsics,
                        reference["support"],
                        condition,
                        float(occluder_profiles[class_name]["center_depth_m"]),
                        float(
                            occluder_profiles[class_name][
                                "nearest_target_support_surface_m"
                            ]
                        ),
                    )
                for _ in range(2):
                    world.tick()
                frame = int(world.tick())
                rgb_image = _wait_exact(queues["rgb"], frame, f"{class_name} {condition} RGB")
                scene_image = _wait_exact(
                    queues["scene_depth"], frame, f"{class_name} {condition} scene depth"
                )
                synchronized = (
                    int(rgb_image.frame) == int(scene_image.frame)
                    and float(rgb_image.timestamp) == float(scene_image.timestamp)
                )
                rgb = _rgb_bgr(rgb_image)
                scene_raw = image_bgra(scene_image)
                scene_depth = decode_depth_bgra(scene_raw)
                metrics = compute_zbuffer_visibility(
                    empty_depth, reference["actor_depth"], scene_depth
                )
                case_name = f"{class_name}_{condition}"
                case_dir = output_dir / "cases" / case_name
                rgb_path = case_dir / "rgb.png"
                support_path = case_dir / "A_i.png"
                visible_path = case_dir / "V_i.png"
                difference_path = case_dir / "depth_difference_metres_f64.npy"
                difference_vis_path = case_dir / "depth_difference_visualization.png"
                case_hashes = {
                    str(rgb_path.relative_to(output_dir)): write_png_x(rgb_path, rgb),
                    str(support_path.relative_to(output_dir)): write_png_x(
                        support_path,
                        np.where(metrics["support"], 255, 0).astype(np.uint8),
                    ),
                    str(visible_path.relative_to(output_dir)): write_png_x(
                        visible_path,
                        np.where(metrics["visible"], 255, 0).astype(np.uint8),
                    ),
                    str(difference_path.relative_to(output_dir)): write_npy_x(
                        difference_path, metrics["depth_difference_m"]
                    ),
                    str(difference_vis_path.relative_to(output_dir)): write_png_x(
                        difference_vis_path,
                        _depth_difference_visualization(
                            metrics["depth_difference_m"], metrics["support"]
                        ),
                    ),
                }
                scene_artifacts = _persist_depth(
                    case_dir, "scene_depth", scene_raw, scene_depth
                )
                case_hashes[
                    str((case_dir / "scene_depth_raw_bgra.png").relative_to(output_dir))
                ] = scene_artifacts["raw_bgra_sha256"]
                case_hashes[
                    str((case_dir / "scene_depth_metres_f64.npy").relative_to(output_dir))
                ] = scene_artifacts["metres_f64_sha256"]
                artifact_hashes.update(case_hashes)
                actor = targets[class_name]
                range_m = float(actor.get_location().distance(normal_camera_transform.location))
                public_metrics = {
                    key: metrics[key]
                    for key in ("support_pixels", "visible_pixels", "visibility")
                }
                provenance = {
                    "schema": "publication_zbuffer_visibility_case_provenance_v2",
                    "case": case_name,
                    "actor_id": int(actor.id),
                    "class_name": class_name,
                    "source_frame": int(scene_image.frame),
                    "source_timestamp": float(scene_image.timestamp),
                    "reference_frame": int(reference["reference_frame"]),
                    "reference_timestamp": float(reference["reference_timestamp"]),
                    "rgb_depth_frame_timestamp_exact": synchronized,
                    "range_m": range_m,
                    "occluder": occluder_record,
                    "tau_empty_m": TAU_EMPTY_M,
                    "tau_match_m": TAU_MATCH_M,
                    "support_source": "isolated actor-only ordinary CARLA depth",
                    "metrics": public_metrics,
                    "artifacts": case_hashes,
                }
                provenance_path = case_dir / "provenance.json"
                provenance_hash = write_json_x(provenance_path, provenance)
                artifact_hashes[str(provenance_path.relative_to(output_dir))] = provenance_hash
                cases[case_name] = {
                    "actor_id": int(actor.id),
                    "class_name": class_name,
                    "condition": condition,
                    "source_frame": int(scene_image.frame),
                    "source_timestamp": float(scene_image.timestamp),
                    "reference_frame": int(reference["reference_frame"]),
                    "synchronized": synchronized,
                    "range_m": range_m,
                    "occluder": occluder_record,
                    "metrics": public_metrics,
                    "artifacts": case_hashes,
                    "rgb": rgb,
                    "scene_depth": scene_depth,
                    "visible": metrics["visible"],
                    "difference": metrics["depth_difference_m"],
                }

        contact_path = output_dir / "controlled_six_case_contact_sheet.png"
        contact_hash = _contact_sheet(cases, references, contact_path)
        artifact_hashes[str(contact_path.relative_to(output_dir))] = contact_hash

        gates = {
            "rgb_normal_depth_frame_timestamp_exact": all(
                bool(case["synchronized"]) for case in cases.values()
            ),
            "empty_reference_has_no_unexpected_nearby_geometry": empty_nearby_pixels == 0,
            "positive_actor_support_both_classes": all(
                int(references[name]["support_pixels"]) > 0 for name in ("vehicle", "person")
            ),
            "reference_intrinsics_and_pixel_coordinates_equal": bool(
                np.array_equal(intrinsics, _intrinsics())
                and WIDTH == int(empty_image.width)
                and HEIGHT == int(empty_image.height)
                and np.array_equal(
                    transform_matrix(normal_camera_transform)[:3, :3],
                    transform_matrix(reference_camera_transform)[:3, :3],
                )
            ),
            "actor_transform_error_le_1e_4": all(
                float(references[name]["transform_error"]) <= TRANSFORM_TOLERANCE
                for name in ("vehicle", "person")
            ),
            "walker_bone_pose_error_le_1e_3": bool(
                references["person"]["walker_bone_error"] is not None
                and float(references["person"]["walker_bone_error"])
                <= WALKER_BONE_TOLERANCE
            ),
            "all_depth_and_visibility_finite_and_bounded": bool(
                np.all(np.isfinite(empty_depth))
                and all(np.all(np.isfinite(row["actor_depth"])) for row in references.values())
                and all(np.all(np.isfinite(case["scene_depth"])) for case in cases.values())
                and all(
                    math.isfinite(float(case["metrics"]["visibility"]))
                    and 0.0 <= float(case["metrics"]["visibility"]) <= 1.0
                    for case in cases.values()
                )
            ),
            "clear_visibility_ge_0_98_both_classes": all(
                float(cases[f"{name}_clear"]["metrics"]["visibility"]) >= 0.98
                for name in ("vehicle", "person")
            ),
            "partial_strictly_between_clear_and_full_both_classes": all(
                float(cases[f"{name}_clear"]["metrics"]["visibility"])
                > float(cases[f"{name}_partial"]["metrics"]["visibility"])
                > float(cases[f"{name}_full"]["metrics"]["visibility"])
                for name in ("vehicle", "person")
            ),
            "full_visibility_le_0_02_both_classes": all(
                float(cases[f"{name}_full"]["metrics"]["visibility"]) <= 0.02
                for name in ("vehicle", "person")
            ),
            "person_support_uses_no_forbidden_proxy": True,
        }
        terminal = QUALIFIED if all(gates.values()) else BLOCKED
        result = {
            "schema": "publication_zbuffer_visibility_controlled_qualification_v2",
            "terminal": terminal,
            "qualified": terminal == QUALIFIED,
            "manual_contact_sheet_inspection_required_before_commit": True,
            "registered_definition": {
                "A_i": "D_actor_i + 0.02 < D_empty",
                "V_i": "A_i and abs(D_scene - D_actor_i) <= 0.02",
                "tau_empty_m": TAU_EMPTY_M,
                "tau_match_m": TAU_MATCH_M,
            },
            "registration": {
                key: registration[key]
                for key in (
                    "lock_sha256",
                    "protocol_sha256",
                    "amendment_sha256",
                    "blocked_evidence_sha256",
                    "previous_failure_sha256",
                    "vehicle_instance_diagnostic_required",
                    "registered_controls_verified",
                )
            },
            "scene": {
                "world": str(world.get_map().name),
                "exact_target_counts": {"vehicle": 1, "person": 1},
                "opaque_occluder_count": 1,
                "opaque_occluder_blueprint": str(occluder.type_id),
                "opaque_occluder_geometry_profiles": occluder_profiles,
                "camera_resolution": [WIDTH, HEIGHT],
                "camera_fov_deg": FOV,
                "camera_intrinsics": intrinsics.tolist(),
                "normal_camera_transform": transform_payload(normal_camera_transform),
                "reference_camera_transform": transform_payload(reference_camera_transform),
            },
            "empty_reference": {
                "frame": int(empty_image.frame),
                "timestamp": float(empty_image.timestamp),
                "nearby_depth_threshold_m": EMPTY_NEARBY_DEPTH_M,
                "nearby_pixels": empty_nearby_pixels,
                "artifacts": empty_artifacts,
            },
            "references": {
                name: {
                    key: value
                    for key, value in row.items()
                    if key not in {"support", "actor_depth"}
                }
                for name, row in references.items()
            },
            "cases": {name: _public_case(case) for name, case in cases.items()},
            "gates": gates,
            "contact_sheet": {
                "path": str(contact_path),
                "sha256": contact_hash,
            },
            "person_support_provenance": {
                "source": "isolated actor-only ordinary CARLA depth",
                "carla_person_instance_or_semantic_pixels_used": False,
                "filled_box_or_ellipse_used": False,
                "learned_mask_used": False,
                "broad_near_far_interval_used": False,
            },
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
        }
        result_path = output_dir / "controlled_qualification_result.json"
        write_json_x(result_path, result)
        marker = output_dir / terminal
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(terminal + "\n")
        return result
    finally:
        for actor in reversed(actors):
            try:
                if str(actor.type_id).startswith("sensor."):
                    actor.stop()
            except RuntimeError:
                pass
            try:
                actor.destroy()
            except RuntimeError:
                pass
        try:
            world.tick()
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
            cleanup["world_settings_restored"] = True
        except RuntimeError:
            cleanup["world_settings_restored"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        result = run(args.host, args.port, output_dir)
    except QualificationBlocked as exc:
        _emit_terminal(output_dir, BLOCKED, f"{type(exc).__name__}: {exc}")
        print(f"{BLOCKED}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 3
    except Exception as exc:
        _emit_terminal(output_dir, IMPLEMENTATION_FAILED, f"{type(exc).__name__}: {exc}")
        print(
            f"{IMPLEMENTATION_FAILED}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 4
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["qualified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

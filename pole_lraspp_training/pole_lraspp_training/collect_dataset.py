from __future__ import annotations

import argparse
import itertools
import math
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .common import (
    append_manifest_rows,
    append_object_box_rows,
    carla_semantic_tags_to_training_mask,
    instance_image_to_tags,
    load_config,
    read_manifest,
    save_json,
    setup_logger,
    stable_split,
    utc_iso,
)

try:
    import carla
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError("CARLA Python bindings are required for data collection.") from exc


def put_latest(q: "queue.Queue[carla.Image]", item: "carla.Image") -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def writer_worker(write_q: "queue.Queue[Optional[Tuple[np.ndarray, Path, List[int]]]]", error_q: "queue.Queue[BaseException]") -> None:
    while True:
        item = write_q.get()
        try:
            if item is None:
                return
            image, path, params = item
            path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(path), image, params)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {path}")
        except BaseException as exc:
            try:
                error_q.put_nowait(exc)
            except queue.Full:
                pass
        finally:
            write_q.task_done()


def raise_writer_errors(error_q: "queue.Queue[BaseException]") -> None:
    try:
        exc = error_q.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"Background image writer failed: {exc}") from exc


def wait_for_frame(q: "queue.Queue[carla.Image]", minimum_frame: int, timeout: float) -> Optional["carla.Image"]:
    deadline = time.time() + float(timeout)
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        try:
            image = q.get(timeout=remaining)
        except queue.Empty:
            return None
        if int(image.frame) >= int(minimum_frame):
            return image


def image_to_bgr(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return np.ascontiguousarray(arr[:, :, :3])


def image_to_bgra(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return np.ascontiguousarray(arr)


def traffic_light_candidates(world: "carla.World") -> List["carla.Actor"]:
    return sorted(world.get_actors().filter("traffic.traffic_light"), key=lambda actor: actor.id)


def traffic_light_opendrive_id(actor: "carla.Actor") -> str:
    try:
        value = actor.get_opendrive_id()
    except Exception:
        return ""
    return "" if value is None else str(value)


def transform_relative_location(base_transform: "carla.Transform", offset: "carla.Location") -> "carla.Location":
    matrix = np.array(base_transform.get_matrix(), dtype=np.float64)
    point = np.array([offset.x, offset.y, offset.z, 1.0], dtype=np.float64)
    x, y, z, _ = matrix @ point
    return carla.Location(x=float(x), y=float(y), z=float(z))


def build_camera_transform(traffic_light: "carla.Actor", cfg: Dict, fov: float, yaw_offset: float, pitch: float) -> "carla.Transform":
    tl_tf = traffic_light.get_transform()
    offset = carla.Location(
        x=float(cfg.get("camera_x_m", 0.0)),
        y=float(cfg.get("camera_y_m", 0.0)),
        z=float(cfg.get("camera_height_m", 6.0)),
    )
    location = transform_relative_location(tl_tf, offset)
    rotation = carla.Rotation(
        pitch=float(pitch),
        yaw=float(tl_tf.rotation.yaw) + float(yaw_offset),
        roll=0.0,
    )
    return carla.Transform(location, rotation)


def fresh_vehicle_blueprint(world: "carla.World", role_name: str) -> "carla.ActorBlueprint":
    blueprints = [
        bp for bp in world.get_blueprint_library().filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels").as_int()) == 4
    ]
    if not blueprints:
        blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    bp = random.choice(blueprints)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    if bp.has_attribute("color"):
        colors = list(bp.get_attribute("color").recommended_values)
        if colors:
            bp.set_attribute("color", random.choice(colors))
    return bp


def spawn_traffic(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    count: int,
    anchor_location: "carla.Location",
    radius: float,
) -> List["carla.Actor"]:
    if count <= 0:
        return []
    spawn_points = world.get_map().get_spawn_points()
    candidates = [sp for sp in spawn_points if sp.location.distance(anchor_location) <= radius]
    if not candidates:
        candidates = spawn_points
    random.shuffle(candidates)
    batch = []
    for idx, sp in enumerate(candidates[: int(count)]):
        command = carla.command.SpawnActor(
            fresh_vehicle_blueprint(world, f"pole_dataset_vehicle_{idx}"),
            sp,
        ).then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port()))
        batch.append(command)
    actors: List["carla.Actor"] = []
    if batch:
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                continue
            actor = world.get_actor(response.actor_id)
            if actor is not None:
                actors.append(actor)
    return actors


def pedestrian_blueprint(world: "carla.World") -> "carla.ActorBlueprint":
    blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    if not blueprints:
        raise RuntimeError("No walker.pedestrian blueprints found.")
    bp = random.choice(blueprints)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")
    return bp


def pedestrian_speed(bp: "carla.ActorBlueprint") -> float:
    if bp.has_attribute("speed"):
        values = list(bp.get_attribute("speed").recommended_values)
        if len(values) >= 2:
            return float(values[1])
        if values:
            return float(values[-1])
    return 1.2


def spawn_pedestrians(
    client: "carla.Client",
    world: "carla.World",
    count: int,
    anchor_location: "carla.Location",
    radius: float,
) -> Tuple[List["carla.Actor"], List["carla.Actor"]]:
    if count <= 0:
        return [], []
    spawn_points = []
    attempts = max(100, int(count) * 25)
    for _ in range(attempts):
        if len(spawn_points) >= count:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if loc.distance(anchor_location) > radius:
            continue
        spawn_points.append(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0)))
    for _ in range(attempts):
        if len(spawn_points) >= count:
            break
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        spawn_points.append(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0)))

    walker_batch = []
    speeds: List[float] = []
    for sp in spawn_points[:count]:
        bp = pedestrian_blueprint(world)
        speeds.append(pedestrian_speed(bp))
        walker_batch.append(carla.command.SpawnActor(bp, sp))

    walker_ids: List[int] = []
    spawned_speeds: List[float] = []
    if walker_batch:
        for response, speed in zip(client.apply_batch_sync(walker_batch, True), speeds):
            if response.error:
                continue
            walker_ids.append(response.actor_id)
            spawned_speeds.append(speed)
    if not walker_ids:
        return [], []

    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    controller_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id) for walker_id in walker_ids]
    controller_ids: List[int] = []
    controller_speeds: List[float] = []
    for response, speed in zip(client.apply_batch_sync(controller_batch, True), spawned_speeds):
        if response.error:
            continue
        controller_ids.append(response.actor_id)
        controller_speeds.append(speed)

    walkers = [world.get_actor(actor_id) for actor_id in walker_ids]
    walkers = [actor for actor in walkers if actor is not None]
    controllers = [world.get_actor(actor_id) for actor_id in controller_ids]
    controllers = [actor for actor in controllers if actor is not None]
    world.tick()
    for controller, speed in zip(controllers, controller_speeds):
        try:
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(float(speed))
        except RuntimeError:
            continue
    return walkers, controllers


def destroy_actors(actors: Sequence["carla.Actor"]) -> None:
    for actor in reversed(list(actors)):
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
        try:
            actor.destroy()
        except RuntimeError:
            pass


def get_camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    K = np.identity(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def bbox_corner_offsets(extent: "carla.Vector3D") -> np.ndarray:
    ex, ey, ez = float(extent.x), float(extent.y), float(extent.z)
    return np.array(
        [
            [ex, ey, ez],
            [ex, ey, -ez],
            [ex, -ey, ez],
            [ex, -ey, -ez],
            [-ex, ey, ez],
            [-ex, ey, -ez],
            [-ex, -ey, ez],
            [-ex, -ey, -ez],
        ],
        dtype=np.float32,
    )


def actor_bbox_world_corners(actor: "carla.Actor") -> Optional[np.ndarray]:
    bbox = getattr(actor, "bounding_box", None)
    if bbox is None:
        return None
    try:
        vertices = bbox.get_world_vertices(actor.get_transform())
        return np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float32)
    except (AttributeError, RuntimeError):
        pass
    try:
        actor_matrix = np.array(actor.get_transform().get_matrix(), dtype=np.float32)
    except RuntimeError:
        return None
    bb_loc = np.array([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float32)
    corners_local = bbox_corner_offsets(bbox.extent) + bb_loc.reshape(1, 3)
    homogeneous = np.concatenate([corners_local, np.ones((8, 1), dtype=np.float32)], axis=1)
    return (actor_matrix @ homogeneous.T).T[:, :3]


def static_bbox_world_corners(bbox: "carla.BoundingBox") -> np.ndarray:
    rotation = getattr(bbox, "rotation", carla.Rotation())
    transform = carla.Transform(bbox.location, rotation)
    matrix = np.array(transform.get_matrix(), dtype=np.float32)
    homogeneous = np.concatenate([bbox_corner_offsets(bbox.extent), np.ones((8, 1), dtype=np.float32)], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def world_to_camera_points(points_world: np.ndarray, cam_inv_matrix: np.ndarray) -> np.ndarray:
    if points_world.size == 0:
        return points_world
    points = np.concatenate([points_world.astype(np.float32), np.ones((len(points_world), 1), dtype=np.float32)], axis=1)
    return (cam_inv_matrix @ points.T).T[:, :3]


def project_camera_points(points_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_cam.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    in_front = x > 0.05
    if not np.any(in_front):
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty
    x2, y2, z2 = x[in_front], y[in_front], z[in_front]
    u = K[0, 2] + (y2 / x2) * K[0, 0]
    v = K[1, 2] - (z2 / x2) * K[1, 1]
    return u, v, x2


def project_corners_to_box(
    corners_world: np.ndarray,
    cam_inv_matrix: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Optional[Dict[str, float]]:
    corners_cam = world_to_camera_points(corners_world, cam_inv_matrix)
    u, v, depths = project_camera_points(corners_cam, K)
    if u.size == 0:
        return None
    u_min = float(np.clip(np.min(u), 0.0, width))
    v_min = float(np.clip(np.min(v), 0.0, height))
    u_max = float(np.clip(np.max(u), 0.0, width))
    v_max = float(np.clip(np.max(v), 0.0, height))
    pixel_w = max(0.0, u_max - u_min)
    pixel_h = max(0.0, v_max - v_min)
    if pixel_w <= 0.0 or pixel_h <= 0.0:
        return None
    return {
        "x": u_min,
        "y": v_min,
        "w": pixel_w,
        "h": pixel_h,
        "area": pixel_w * pixel_h,
        "center_x": u_min + pixel_w / 2.0,
        "center_y": v_min + pixel_h / 2.0,
        "depth": float(np.min(depths)),
    }


def object_extent_fields(extent: object) -> Dict[str, float]:
    x = float(getattr(extent, "x", 0.0))
    y = float(getattr(extent, "y", 0.0))
    z = float(getattr(extent, "z", 0.0))
    return {
        "gt_extent_x_m": x,
        "gt_extent_y_m": y,
        "gt_extent_z_m": z,
        "gt_size_x_m": x * 2.0,
        "gt_size_y_m": y * 2.0,
        "gt_size_z_m": z * 2.0,
    }


def iter_static_level_bboxes(world: "carla.World") -> Sequence[Tuple[str, str, "carla.BoundingBox"]]:
    if not hasattr(world, "get_level_bbs"):
        return []
    items: List[Tuple[str, str, "carla.BoundingBox"]] = []
    for label, carla_label_name in (("vehicle", "Vehicles"), ("person", "Pedestrians")):
        carla_label = getattr(carla.CityObjectLabel, carla_label_name, None)
        if carla_label is None:
            continue
        try:
            for bbox in world.get_level_bbs(carla_label):
                items.append((label, "level_bbox", bbox))
        except Exception:
            continue
    return items


def project_ground_truth_boxes(
    world: "carla.World",
    camera_actor: "carla.Actor",
    *,
    sample_id: str,
    frame_id: int,
    timestamp: str,
    experiment_id: str,
    traffic_light_id: int,
    scenario_id: int,
    view_id: str,
    width: int,
    height: int,
    fov: float,
    max_distance_m: float,
    include_static_level_bboxes: bool,
) -> List[Dict]:
    try:
        camera_transform = camera_actor.get_transform()
        camera_location = camera_transform.location
        cam_inv_matrix = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)
    except RuntimeError:
        return []
    K = get_camera_intrinsics(width, height, fov)
    common = {
        "experiment_id": experiment_id,
        "sample_id": sample_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "traffic_light_id": traffic_light_id,
        "scenario_id": scenario_id,
        "view_id": view_id,
    }
    rows: List[Dict] = []
    max_distance_m = max(0.0, float(max_distance_m))

    for label, pattern in (("vehicle", "vehicle.*"), ("person", "walker.pedestrian.*")):
        for actor in world.get_actors().filter(pattern):
            try:
                distance = float(actor.get_location().distance(camera_location))
            except RuntimeError:
                continue
            if max_distance_m > 0.0 and distance > max_distance_m:
                continue
            corners = actor_bbox_world_corners(actor)
            if corners is None:
                continue
            box = project_corners_to_box(corners, cam_inv_matrix, K, width, height)
            if box is None:
                continue
            bbox = getattr(actor, "bounding_box", None)
            extent = getattr(bbox, "extent", None) if bbox is not None else None
            row = dict(common)
            row.update(
                {
                    "label": label,
                    "gt_actor_id": str(actor.id),
                    "gt_source": "actor",
                    "gt_actor_type_id": str(actor.type_id),
                    "gt_bbox_x": box["x"],
                    "gt_bbox_y": box["y"],
                    "gt_bbox_w": box["w"],
                    "gt_bbox_h": box["h"],
                    "gt_bbox_area_px": box["area"],
                    "gt_center_x": box["center_x"],
                    "gt_center_y": box["center_y"],
                    "gt_depth_m": box["depth"],
                    "gt_distance_m": distance,
                }
            )
            row.update(object_extent_fields(extent))
            rows.append(row)

    if include_static_level_bboxes:
        for source_index, (label, source_name, bbox) in enumerate(iter_static_level_bboxes(world)):
            distance = float(bbox.location.distance(camera_location))
            if max_distance_m > 0.0 and distance > max_distance_m:
                continue
            box = project_corners_to_box(static_bbox_world_corners(bbox), cam_inv_matrix, K, width, height)
            if box is None:
                continue
            row = dict(common)
            row.update(
                {
                    "label": label,
                    "gt_actor_id": f"{source_name}_{source_index}",
                    "gt_source": source_name,
                    "gt_actor_type_id": f"static.{label}",
                    "gt_bbox_x": box["x"],
                    "gt_bbox_y": box["y"],
                    "gt_bbox_w": box["w"],
                    "gt_bbox_h": box["h"],
                    "gt_bbox_area_px": box["area"],
                    "gt_center_x": box["center_x"],
                    "gt_center_y": box["center_y"],
                    "gt_depth_m": box["depth"],
                    "gt_distance_m": distance,
                }
            )
            row.update(object_extent_fields(bbox.extent))
            rows.append(row)
    return rows


def camera_bp(world: "carla.World", bp_id: str, width: int, height: int, fov: float, fps: float) -> "carla.ActorBlueprint":
    bp = world.get_blueprint_library().find(bp_id)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(float(fov)))
    bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(fps))))
    return bp


def scenario_schedule(config: Dict, traffic_lights: Sequence["carla.Actor"]) -> List[Dict]:
    c = config["collection"]
    combos = list(itertools.product(
        c.get("traffic_densities", [20]),
        c.get("pedestrian_densities", [0]),
        c.get("fovs", [100.0]),
        c.get("yaw_offsets", [70.0]),
        c.get("pitches", [-35.0]),
    ))
    random.Random(int(c.get("seed", 17))).shuffle(combos)
    schedule: List[Dict] = []
    for tl in traffic_lights:
        for idx, (vehicles, peds, fov, yaw, pitch) in enumerate(combos):
            schedule.append(
                {
                    "traffic_light_id": int(tl.id),
                    "scenario_index": idx,
                    "traffic_density": int(vehicles),
                    "pedestrian_density": int(peds),
                    "fov": float(fov),
                    "yaw_offset": float(yaw),
                    "pitch": float(pitch),
                }
            )
    return schedule


def collect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    data_dir = exp_dir / "dataset"
    rgb_dir = data_dir / "rgb"
    mask_dir = data_dir / "mask_3class"
    raw_dir = data_dir / "instance_raw"
    for directory in (rgb_dir, mask_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.csv"
    object_boxes_path = data_dir / "object_boxes.csv"
    state_path = data_dir / "collection_state.json"
    log = setup_logger(exp_dir / "supervisor.log")

    set_seed = int(config["collection"].get("seed", 17))
    random.seed(set_seed)
    np.random.seed(set_seed)

    client = carla.Client(config["carla"].get("host", "127.0.0.1"), int(config["carla"].get("port", 2000)))
    client.set_timeout(20.0)
    world = client.load_world(config["carla"].get("town")) if str(config["carla"].get("town", "")).strip() else client.get_world()
    traffic_manager = client.get_trafficmanager(int(config["carla"].get("tm_port", 8000)))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    try:
        traffic_manager.set_random_device_seed(set_seed)
        world.set_pedestrians_seed(set_seed)
    except Exception:
        pass

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / max(0.1, float(config["collection"].get("fps", 10.0)))
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    world.tick()

    traffic_lights = traffic_light_candidates(world)
    if not traffic_lights:
        raise RuntimeError("No traffic lights found in current CARLA world.")
    log(f"Discovered {len(traffic_lights)} traffic lights in {world.get_map().name}.")
    save_json(data_dir / "traffic_lights.json", {
        "map": world.get_map().name,
        "traffic_lights": [
            {
                "id": int(tl.id),
                "opendrive_id": traffic_light_opendrive_id(tl),
                "transform": str(tl.get_transform()),
            }
            for tl in traffic_lights
        ],
    })

    completed_ids = {row["sample_id"] for row in read_manifest(manifest_path)}
    schedule = scenario_schedule(config, traffic_lights)
    collection_deadline = time.monotonic() + float(config.get("collection_budget_hours", 2.5)) * 3600.0
    max_samples = int(config["collection"].get("max_samples", 0) or 0)
    width, height = [int(v) for v in config["collection"].get("resolution", [854, 480])]
    fps = float(config["collection"].get("fps", 10.0))
    frames_per_view = int(config["collection"].get("frames_per_view", 36))
    jpeg_quality = int(config["collection"].get("jpeg_quality", 92))
    split_cfg = config.get("splits", {})
    split_seed = int(split_cfg.get("seed", 23))
    radius = float(config["collection"].get("spawn_radius_m", 110.0))
    rows_written = 0
    actors: List["carla.Actor"] = []
    write_q: "queue.Queue[Optional[Tuple[np.ndarray, Path, List[int]]]]" = queue.Queue(maxsize=int(config["collection"].get("writer_queue_size", 64)))
    writer_errors: "queue.Queue[BaseException]" = queue.Queue(maxsize=1)
    writer_thread = threading.Thread(target=writer_worker, args=(write_q, writer_errors), daemon=True)
    writer_thread.start()

    try:
        for scenario_id, item in enumerate(schedule):
            if time.monotonic() >= collection_deadline:
                log("Collection budget exhausted; stopping collection stage.")
                break
            if max_samples > 0 and len(completed_ids) >= max_samples:
                log(f"Reached max_samples={max_samples}; stopping collection stage.")
                break

            tl = next((candidate for candidate in traffic_lights if int(candidate.id) == int(item["traffic_light_id"])), None)
            if tl is None:
                continue
            scenario_key = (
                f"tl{tl.id}_sc{item['scenario_index']}_veh{item['traffic_density']}"
                f"_ped{item['pedestrian_density']}_fov{item['fov']}_yaw{item['yaw_offset']}_p{item['pitch']}"
            )
            if all(f"{scenario_key}_frame{frame_idx:04d}" in completed_ids for frame_idx in range(frames_per_view)):
                continue

            anchor = tl.get_transform().location
            actors = []
            actors.extend(spawn_traffic(client, world, traffic_manager, item["traffic_density"], anchor, radius))
            walkers, controllers = spawn_pedestrians(client, world, item["pedestrian_density"], anchor, radius)
            actors.extend(walkers)
            actors.extend(controllers)
            world.tick()

            rgb_q: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
            inst_q: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
            transform = build_camera_transform(tl, config["collection"], item["fov"], item["yaw_offset"], item["pitch"])
            rgb = world.spawn_actor(camera_bp(world, "sensor.camera.rgb", width, height, item["fov"], fps), transform)
            inst = world.spawn_actor(camera_bp(world, "sensor.camera.instance_segmentation", width, height, item["fov"], fps), transform)
            actors.extend([rgb, inst])
            rgb.listen(lambda image, q=rgb_q: put_latest(q, image))
            inst.listen(lambda image, q=inst_q: put_latest(q, image))

            for _ in range(int(config["collection"].get("warmup_ticks", 10))):
                world.tick()
                time.sleep(0.001)

            rows: List[Dict] = []
            object_box_rows: List[Dict] = []
            K = get_camera_intrinsics(width, height, float(item["fov"]))
            for frame_idx in range(frames_per_view):
                raise_writer_errors(writer_errors)
                if time.monotonic() >= collection_deadline:
                    break
                sample_id = f"{scenario_key}_frame{frame_idx:04d}"
                if sample_id in completed_ids:
                    world.tick()
                    continue
                frame = int(world.tick())
                rgb_image = wait_for_frame(rgb_q, frame, 5.0)
                inst_image = wait_for_frame(inst_q, frame, 5.0)
                if rgb_image is None or inst_image is None:
                    raise RuntimeError(f"Timed out waiting for paired camera frames at world frame {frame}.")

                bgr = image_to_bgr(rgb_image)
                raw_bgra = image_to_bgra(inst_image)
                tags = instance_image_to_tags(raw_bgra)
                mask = carla_semantic_tags_to_training_mask(tags)
                timestamp = utc_iso()

                rgb_rel = Path("rgb") / f"{sample_id}.jpg"
                mask_rel = Path("mask_3class") / f"{sample_id}.png"
                raw_rel = Path("instance_raw") / f"{sample_id}.png"
                write_q.put((bgr, data_dir / rgb_rel, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]))
                write_q.put((mask, data_dir / mask_rel, []))
                write_q.put((raw_bgra, data_dir / raw_rel, []))

                split = stable_split(sample_id, split_cfg, split_seed)
                row = {
                    "experiment_id": exp_dir.name,
                    "sample_id": sample_id,
                    "split": split,
                    "rgb_path": str(rgb_rel),
                    "mask_path": str(mask_rel),
                    "instance_raw_path": str(raw_rel),
                    "frame_id": int(rgb_image.frame),
                    "timestamp": timestamp,
                    "traffic_light_id": int(tl.id),
                    "traffic_light_opendrive_id": traffic_light_opendrive_id(tl),
                    "map_name": world.get_map().name,
                    "camera_x": float(transform.location.x),
                    "camera_y": float(transform.location.y),
                    "camera_z": float(transform.location.z),
                    "camera_pitch": float(transform.rotation.pitch),
                    "camera_yaw": float(transform.rotation.yaw),
                    "camera_roll": float(transform.rotation.roll),
                    "camera_fov": float(item["fov"]),
                    "camera_width": width,
                    "camera_height": height,
                    "camera_fx": float(K[0, 0]),
                    "camera_fy": float(K[1, 1]),
                    "camera_cx": float(K[0, 2]),
                    "camera_cy": float(K[1, 2]),
                    "traffic_density": int(item["traffic_density"]),
                    "pedestrian_density": int(item["pedestrian_density"]),
                    "scenario_id": scenario_id,
                    "view_id": scenario_key,
                    "vehicle_pixels": int(np.sum(mask == 1)),
                    "person_pixels": int(np.sum(mask == 2)),
                }
                rows.append(row)
                object_box_rows.extend(
                    project_ground_truth_boxes(
                        world,
                        rgb,
                        sample_id=sample_id,
                        frame_id=int(rgb_image.frame),
                        timestamp=timestamp,
                        experiment_id=exp_dir.name,
                        traffic_light_id=int(tl.id),
                        scenario_id=int(scenario_id),
                        view_id=scenario_key,
                        width=width,
                        height=height,
                        fov=float(item["fov"]),
                        max_distance_m=float(config["collection"].get("gt_bbox_max_distance_m", 140.0)),
                        include_static_level_bboxes=bool(config["collection"].get("include_static_level_bboxes", True)),
                    )
                )
                completed_ids.add(sample_id)
                rows_written += 1
                if max_samples > 0 and len(completed_ids) >= max_samples:
                    break
            if rows:
                write_q.join()
                raise_writer_errors(writer_errors)
                append_manifest_rows(manifest_path, rows)
                if object_box_rows:
                    append_object_box_rows(object_boxes_path, object_box_rows)
                save_json(state_path, {
                    "updated_at": utc_iso(),
                    "samples": len(completed_ids),
                    "last_scenario": scenario_key,
                    "rows_written_this_process": rows_written,
                    "object_boxes_path": str(object_boxes_path),
                })
                log(f"Collected {len(rows)} rows for {scenario_key}; total={len(completed_ids)}.")
            destroy_actors(actors)
            actors = []
    finally:
        try:
            write_q.put(None)
            write_q.join()
        except Exception:
            pass
        destroy_actors(actors)
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

    save_json(data_dir / "manifest.json", {
        "experiment_id": exp_dir.name,
        "updated_at": utc_iso(),
        "manifest_csv": str(manifest_path),
        "object_boxes_csv": str(object_boxes_path),
        "samples": len(completed_ids),
        "rgb_dir": str(rgb_dir),
        "mask_dir": str(mask_dir),
        "instance_raw_dir": str(raw_dir),
        "traffic_lights_json": str(data_dir / "traffic_lights.json"),
    })
    log(f"Collection finished with {len(completed_ids)} samples in {manifest_path}.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(collect(args))


if __name__ == "__main__":
    main()

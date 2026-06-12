from __future__ import annotations

import argparse
import itertools
import json
import queue
import random
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
from .radar_fusion import (
    StationaryTrackAccumulator,
    build_radar_sample,
    radar_raw_to_alt_az_depth_velocity,
)

try:
    import carla
except Exception as exc:  # pragma: no cover
    raise RuntimeError("CARLA Python bindings are required for data collection.") from exc

from pole_lraspp_training.collect_dataset import (  # type: ignore
    build_camera_transform,
    camera_bp,
    destroy_actors,
    get_camera_intrinsics,
    image_to_bgr,
    image_to_bgra,
    project_ground_truth_boxes,
    put_latest,
    raise_writer_errors,
    spawn_pedestrians,
    spawn_traffic,
    traffic_light_candidates,
    traffic_light_opendrive_id,
    wait_for_frame,
    writer_worker,
)


def matrix_json(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix, dtype=np.float64).tolist(), separators=(",", ":"))


def multimodal_scenario_schedule(config: Dict, traffic_lights: Sequence["carla.Actor"]) -> List[Dict]:
    c = config["collection"]
    combos = list(
        itertools.product(
            c.get("traffic_densities", [20]),
            c.get("pedestrian_densities", [0]),
            c.get("fovs", [100.0]),
            c.get("yaw_offsets", [70.0]),
            c.get("pitches", [-35.0]),
        )
    )
    if not combos:
        return []
    seed = int(c.get("seed", 17))
    scenarios_per_anchor = int(c.get("scenarios_per_anchor", 0) or 0)
    if scenarios_per_anchor <= 0:
        scenarios_per_anchor = len(combos)
    per_anchor: Dict[int, List[Tuple[int, Tuple[object, ...]]]] = {}
    for tl in traffic_lights:
        indexed = list(enumerate(combos))
        random.Random(seed + int(tl.id) * 1009).shuffle(indexed)
        per_anchor[int(tl.id)] = indexed[: min(scenarios_per_anchor, len(indexed))]
    schedule: List[Dict] = []
    for slot in range(scenarios_per_anchor):
        for tl in traffic_lights:
            tl_items = per_anchor.get(int(tl.id), [])
            if slot >= len(tl_items):
                continue
            combo_index, (vehicles, peds, fov, yaw, pitch) = tl_items[slot]
            schedule.append(
                {
                    "traffic_light_id": int(tl.id),
                    "scenario_index": int(combo_index),
                    "traffic_density": int(vehicles),
                    "pedestrian_density": int(peds),
                    "fov": float(fov),
                    "yaw_offset": float(yaw),
                    "pitch": float(pitch),
                }
            )
    return schedule


def put_latest_radar(
    q: "queue.Queue[Tuple[int, np.ndarray, float]]",
    frame_id: int,
    detections: np.ndarray,
    timestamp: float,
) -> None:
    item = (int(frame_id), detections.astype(np.float32, copy=False), float(timestamp))
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def wait_for_radar(
    q: "queue.Queue[Tuple[int, np.ndarray, float]]",
    minimum_frame: int,
    timeout: float,
) -> Optional[Tuple[int, np.ndarray, float]]:
    deadline = time.time() + float(timeout)
    best: Optional[Tuple[int, np.ndarray, float]] = None
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return best
        try:
            item = q.get(timeout=remaining)
        except queue.Empty:
            return best
        if int(item[0]) >= int(minimum_frame):
            return item
        best = item


def radar_bp(world: "carla.World", cfg: Dict, camera_fov: float, fps: float) -> "carla.ActorBlueprint":
    radar_cfg = cfg.get("radar", {})
    bp = world.get_blueprint_library().find("sensor.other.radar")
    horizontal_fov = float(radar_cfg.get("horizontal_fov", camera_fov))
    if bool(radar_cfg.get("match_camera_fov", True)):
        horizontal_fov = float(camera_fov)
    bp.set_attribute("horizontal_fov", str(horizontal_fov))
    bp.set_attribute("vertical_fov", str(float(radar_cfg.get("vertical_fov", 30.0))))
    bp.set_attribute("range", str(float(radar_cfg.get("range_m", 120.0))))
    bp.set_attribute("points_per_second", str(int(radar_cfg.get("points_per_second", 5000))))
    bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(fps))))
    return bp


def _transform_world_to_sensor(camera_inverse_matrix: np.ndarray, location: "carla.Location") -> np.ndarray:
    point = np.array([float(location.x), float(location.y), float(location.z), 1.0], dtype=np.float64)
    return (camera_inverse_matrix @ point)[:3]


def _update_stationary_age(
    tracks: Dict[str, Dict[str, float]],
    actor_id: str,
    speed_mps: float,
    frame_time_s: float,
    *,
    stationary_velocity_mps: float,
    parked_threshold_s: float,
) -> float:
    now = float(frame_time_s)
    track = tracks.get(actor_id, {"age_s": 0.0, "last_seen_s": now})
    dt = max(0.0, now - float(track.get("last_seen_s", now)))
    if float(speed_mps) <= float(stationary_velocity_mps):
        track["age_s"] = min(float(parked_threshold_s) * 3.0, float(track.get("age_s", 0.0)) + dt)
    else:
        track["age_s"] = 0.0
    track["last_seen_s"] = now
    tracks[actor_id] = track
    return float(track["age_s"])


def _radar_support_count(row: Dict, radar_points: Dict[str, np.ndarray]) -> int:
    if not radar_points:
        return 0
    try:
        u = radar_points["u"]
        v = radar_points["v"]
        valid = radar_points["valid_projection"].astype(bool)
    except KeyError:
        return 0
    x0 = float(row.get("gt_bbox_x", 0.0) or 0.0)
    y0 = float(row.get("gt_bbox_y", 0.0) or 0.0)
    x1 = x0 + float(row.get("gt_bbox_w", 0.0) or 0.0)
    y1 = y0 + float(row.get("gt_bbox_h", 0.0) or 0.0)
    inside = valid & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    return int(np.sum(inside))


def actor_global_fields(
    world: "carla.World",
    rows: Sequence[Dict],
    *,
    camera_inverse_matrix: np.ndarray,
    frame_time_s: float,
    actor_stationary_tracks: Dict[str, Dict[str, float]],
    stationary_velocity_mps: float,
    parked_threshold_s: float,
    radar_points: Dict[str, np.ndarray],
) -> List[Dict]:
    output: List[Dict] = []
    for row in rows:
        updated = dict(row)
        if str(row.get("gt_source", "")) == "actor":
            try:
                actor = world.get_actor(int(str(row.get("gt_actor_id", ""))))
            except Exception:
                actor = None
            if actor is not None:
                try:
                    loc = actor.get_location()
                    vel = actor.get_velocity()
                    sensor_xyz = _transform_world_to_sensor(camera_inverse_matrix, loc)
                    yaw = float(actor.get_transform().rotation.yaw)
                    speed_mps = float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5)
                    actor_id = str(row.get("gt_actor_id", ""))
                    stationary_age = _update_stationary_age(
                        actor_stationary_tracks,
                        actor_id,
                        speed_mps,
                        frame_time_s,
                        stationary_velocity_mps=stationary_velocity_mps,
                        parked_threshold_s=parked_threshold_s,
                    )
                    updated["object_world_x"] = float(loc.x)
                    updated["object_world_y"] = float(loc.y)
                    updated["object_world_z"] = float(loc.z)
                    updated["object_sensor_x"] = float(sensor_xyz[0])
                    updated["object_sensor_y"] = float(sensor_xyz[1])
                    updated["object_sensor_z"] = float(sensor_xyz[2])
                    updated["object_yaw_deg"] = yaw
                    updated["object_velocity_x_mps"] = float(vel.x)
                    updated["object_velocity_y_mps"] = float(vel.y)
                    updated["object_velocity_z_mps"] = float(vel.z)
                    updated["object_speed_mps"] = speed_mps
                    updated["stationary_age_s"] = stationary_age
                    updated["stationary_label"] = int(speed_mps <= float(stationary_velocity_mps))
                    updated["parked_label"] = int(stationary_age >= float(parked_threshold_s))
                    updated["radar_support_points"] = _radar_support_count(updated, radar_points)
                except RuntimeError:
                    pass
        output.append(updated)
    return output


def collect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    data_dir = exp_dir / "dataset"
    rgb_dir = data_dir / "rgb"
    mask_dir = data_dir / "mask_3class"
    raw_dir = data_dir / "instance_raw"
    radar_tensor_dir = data_dir / "radar_tensor"
    radar_points_dir = data_dir / "radar_points"
    for directory in (rgb_dir, mask_dir, raw_dir, radar_tensor_dir, radar_points_dir):
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
    save_json(
        data_dir / "traffic_lights.json",
        {
            "map": world.get_map().name,
            "traffic_lights": [
                {
                    "id": int(tl.id),
                    "opendrive_id": traffic_light_opendrive_id(tl),
                    "transform": str(tl.get_transform()),
                    "matrix": np.array(tl.get_transform().get_matrix(), dtype=np.float64).tolist(),
                }
                for tl in traffic_lights
            ],
        },
    )

    completed_ids = {row["sample_id"] for row in read_manifest(manifest_path)}
    schedule = multimodal_scenario_schedule(config, traffic_lights)
    collection_deadline = time.monotonic() + float(config.get("collection_budget_hours", 2.5)) * 3600.0
    max_samples = int(config["collection"].get("max_samples", 0) or 0)
    width, height = [int(v) for v in config["collection"].get("resolution", [854, 480])]
    fps = float(config["collection"].get("fps", 10.0))
    frames_per_view = int(config["collection"].get("frames_per_view", 36))
    jpeg_quality = int(config["collection"].get("jpeg_quality", 92))
    split_cfg = config.get("splits", {})
    split_seed = int(split_cfg.get("seed", 23))
    radius = float(config["collection"].get("spawn_radius_m", 110.0))
    radar_cfg = config["collection"].get("radar", {})
    stationary_cfg = config.get("fusion", {}).get("stationary_tracker", {})
    require_all_anchors = bool(config["collection"].get("require_all_anchors", False))
    rows_written = 0
    actors: List["carla.Actor"] = []
    write_q: "queue.Queue[Optional[Tuple[np.ndarray, Path, List[int]]]]" = queue.Queue(
        maxsize=int(config["collection"].get("writer_queue_size", 64))
    )
    writer_errors: "queue.Queue[BaseException]" = queue.Queue(maxsize=1)
    writer_thread = threading.Thread(target=writer_worker, args=(write_q, writer_errors), daemon=True)
    writer_thread.start()

    try:
        for scenario_id, item in enumerate(schedule):
            if time.monotonic() >= collection_deadline and not require_all_anchors:
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
            radar_q: "queue.Queue[Tuple[int, np.ndarray, float]]" = queue.Queue(maxsize=3)
            transform = build_camera_transform(tl, config["collection"], item["fov"], item["yaw_offset"], item["pitch"])
            rgb = world.spawn_actor(camera_bp(world, "sensor.camera.rgb", width, height, item["fov"], fps), transform)
            inst = world.spawn_actor(camera_bp(world, "sensor.camera.instance_segmentation", width, height, item["fov"], fps), transform)
            radar = world.spawn_actor(radar_bp(world, config["collection"], item["fov"], fps), transform)
            actors.extend([rgb, inst, radar])
            rgb.listen(lambda image, q=rgb_q: put_latest(q, image))
            inst.listen(lambda image, q=inst_q: put_latest(q, image))

            def _radar_callback(measurement: "carla.RadarMeasurement") -> None:
                put_latest_radar(
                    radar_q,
                    int(measurement.frame),
                    radar_raw_to_alt_az_depth_velocity(measurement.raw_data),
                    float(measurement.timestamp),
                )

            radar.listen(_radar_callback)

            tracker = StationaryTrackAccumulator(
                stationary_velocity_mps=float(stationary_cfg.get("stationary_velocity_mps", 0.35)),
                parked_threshold_s=float(stationary_cfg.get("parked_threshold_s", 5.0)),
                association_grid_m=float(stationary_cfg.get("association_grid_m", 1.5)),
                max_stale_s=float(stationary_cfg.get("max_stale_s", 2.0)),
            )
            actor_stationary_tracks: Dict[str, Dict[str, float]] = {}
            for _ in range(int(config["collection"].get("warmup_ticks", 10))):
                world.tick()
                time.sleep(0.001)

            rows: List[Dict] = []
            object_box_rows: List[Dict] = []
            K = get_camera_intrinsics(width, height, float(item["fov"]))
            sensor_matrix = np.array(transform.get_matrix(), dtype=np.float64)
            camera_inverse_matrix = np.array(transform.get_inverse_matrix(), dtype=np.float64)
            radar_matrix = np.array(radar.get_transform().get_matrix(), dtype=np.float64)
            radar_inverse_matrix = np.array(radar.get_transform().get_inverse_matrix(), dtype=np.float64)
            radar_to_camera_matrix = camera_inverse_matrix @ radar_matrix
            anchor_tf = tl.get_transform()
            for frame_idx in range(frames_per_view):
                raise_writer_errors(writer_errors)
                if time.monotonic() >= collection_deadline and not require_all_anchors:
                    break
                sample_id = f"{scenario_key}_frame{frame_idx:04d}"
                if sample_id in completed_ids:
                    world.tick()
                    continue
                frame = int(world.tick())
                rgb_image = wait_for_frame(rgb_q, frame, 5.0)
                inst_image = wait_for_frame(inst_q, frame, 5.0)
                radar_item = wait_for_radar(radar_q, frame, 5.0)
                if rgb_image is None or inst_image is None or radar_item is None:
                    raise RuntimeError(f"Timed out waiting for RGB/instance/radar frames at world frame {frame}.")

                radar_frame, detections, radar_timestamp = radar_item
                bgr = image_to_bgr(rgb_image)
                raw_bgra = image_to_bgra(inst_image)
                tags = instance_image_to_tags(raw_bgra)
                mask = carla_semantic_tags_to_training_mask(tags)
                timestamp = utc_iso()
                radar_tensor, radar_points, radar_summary = build_radar_sample(
                    detections=detections,
                    sensor_matrix=sensor_matrix,
                    camera_inverse_matrix=camera_inverse_matrix,
                    camera_intrinsics=K,
                    width=width,
                    height=height,
                    frame_time_s=float(radar_timestamp),
                    tracker=tracker,
                    max_range_m=float(radar_cfg.get("range_m", 120.0)),
                    max_abs_velocity_mps=float(radar_cfg.get("max_abs_velocity_mps", 20.0)),
                    parked_threshold_s=float(stationary_cfg.get("parked_threshold_s", 5.0)),
                    point_radius_px=int(radar_cfg.get("raster_point_radius_px", 2)),
                )

                rgb_rel = Path("rgb") / f"{sample_id}.jpg"
                mask_rel = Path("mask_3class") / f"{sample_id}.png"
                raw_rel = Path("instance_raw") / f"{sample_id}.png"
                radar_tensor_rel = Path("radar_tensor") / f"{sample_id}.npz"
                radar_points_rel = Path("radar_points") / f"{sample_id}.npz"
                write_q.put((bgr, data_dir / rgb_rel, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]))
                write_q.put((mask, data_dir / mask_rel, []))
                write_q.put((raw_bgra, data_dir / raw_rel, []))
                np.savez_compressed(data_dir / radar_tensor_rel, radar=radar_tensor)
                np.savez_compressed(data_dir / radar_points_rel, **radar_points)

                split = stable_split(sample_id, split_cfg, split_seed)
                row = {
                    "experiment_id": exp_dir.name,
                    "sample_id": sample_id,
                    "split": split,
                    "rgb_path": str(rgb_rel),
                    "mask_path": str(mask_rel),
                    "instance_raw_path": str(raw_rel),
                    "radar_tensor_path": str(radar_tensor_rel),
                    "radar_points_path": str(radar_points_rel),
                    "frame_id": int(rgb_image.frame),
                    "radar_frame_id": int(radar_frame),
                    "timestamp": timestamp,
                    "radar_timestamp": float(radar_timestamp),
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
                    "camera_matrix_json": matrix_json(sensor_matrix),
                    "camera_inverse_matrix_json": matrix_json(camera_inverse_matrix),
                    "radar_matrix_json": matrix_json(radar_matrix),
                    "radar_inverse_matrix_json": matrix_json(radar_inverse_matrix),
                    "radar_to_camera_matrix_json": matrix_json(radar_to_camera_matrix),
                    "anchor_x": float(anchor_tf.location.x),
                    "anchor_y": float(anchor_tf.location.y),
                    "anchor_z": float(anchor_tf.location.z),
                    "anchor_pitch": float(anchor_tf.rotation.pitch),
                    "anchor_yaw": float(anchor_tf.rotation.yaw),
                    "anchor_roll": float(anchor_tf.rotation.roll),
                    "radar_horizontal_fov": float(item["fov"]) if bool(radar_cfg.get("match_camera_fov", True)) else float(radar_cfg.get("horizontal_fov", item["fov"])),
                    "radar_vertical_fov": float(radar_cfg.get("vertical_fov", 30.0)),
                    "radar_range_m": float(radar_cfg.get("range_m", 120.0)),
                    "radar_points": int(radar_summary["radar_points"]),
                    "radar_stationary_points": int(radar_summary["radar_stationary_points"]),
                    "radar_parked_evidence_points": int(radar_summary["radar_parked_evidence_points"]),
                    "traffic_density": int(item["traffic_density"]),
                    "pedestrian_density": int(item["pedestrian_density"]),
                    "scenario_id": scenario_id,
                    "view_id": scenario_key,
                    "vehicle_pixels": int(np.sum(mask == 1)),
                    "person_pixels": int(np.sum(mask == 2)),
                }
                rows.append(row)
                object_box_rows.extend(
                    actor_global_fields(
                        world,
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
                        ),
                        camera_inverse_matrix=camera_inverse_matrix,
                        frame_time_s=float(radar_timestamp),
                        actor_stationary_tracks=actor_stationary_tracks,
                        stationary_velocity_mps=float(stationary_cfg.get("stationary_velocity_mps", 0.35)),
                        parked_threshold_s=float(stationary_cfg.get("parked_threshold_s", 5.0)),
                        radar_points=radar_points,
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
                save_json(
                    state_path,
                    {
                        "updated_at": utc_iso(),
                        "samples": len(completed_ids),
                        "last_scenario": scenario_key,
                        "rows_written_this_process": rows_written,
                        "object_boxes_path": str(object_boxes_path),
                    },
                )
                log(f"Collected {len(rows)} multimodal rows for {scenario_key}; total={len(completed_ids)}.")
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

    save_json(
        data_dir / "manifest.json",
        {
            "experiment_id": exp_dir.name,
            "updated_at": utc_iso(),
            "manifest_csv": str(manifest_path),
            "object_boxes_csv": str(object_boxes_path),
            "samples": len(completed_ids),
            "rgb_dir": str(rgb_dir),
            "mask_dir": str(mask_dir),
            "instance_raw_dir": str(raw_dir),
            "radar_tensor_dir": str(radar_tensor_dir),
            "radar_points_dir": str(radar_points_dir),
            "traffic_lights_json": str(data_dir / "traffic_lights.json"),
        },
    )
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

#!/usr/bin/env python3
"""Curbside raw-vs-semantic LiDAR diagnostic.

This is a dedicated copy-path for the curbside accident scenario. The original
curbside demo harness is left untouched; this runner imports its layout and
target-crossing helpers, then attaches paired raw/semantic LiDAR sensors to the
ego vehicle so we can measure pedestrian and vehicle localization under a
deterministic crossing event.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import carla
except ImportError:  # pragma: no cover - lab CARLA env provides this.
    carla = None

import carla_lidar_raw_vs_semantic_diagnostic as lidar_diag
from scenesense_scenarios import scenesense_scenario_harness as harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic curbside pedestrian crossing with paired raw/semantic LiDAR."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--load-town", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--output-root", default="lidar_diagnostic_runs")
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--warmup-ticks", type=int, default=15)
    parser.add_argument("--asynch", action="store_true")

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-width", type=int, default=1440)
    parser.add_argument("--preview-height", type=int, default=810)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=120.0)

    parser.add_argument("--anchor-spawn-index", type=int, default=152)
    parser.add_argument("--ego-spawn-index", type=int, default=152)
    parser.add_argument(
        "--ego-motion",
        choices=("stationary", "drive"),
        default="stationary",
        help=(
            "stationary keeps the ego parked and starts the pedestrian by delay only; "
            "drive recreates the curbside collision-style ego motion."
        ),
    )
    parser.add_argument("--ego-target-speed", type=float, default=15.2)
    parser.add_argument("--ego-drive-throttle", type=float, default=0.45)
    parser.add_argument("--ego-route-lookahead-m", type=float, default=24.0)
    parser.add_argument("--target-crossing-delay-s", type=float, default=3.0)
    parser.add_argument("--target-crossing-speed", type=float, default=26.5)
    parser.add_argument("--target-crossing-control-speed", type=float, default=26.5)
    parser.add_argument("--target-crossing-trigger-route-lead-m", type=float, default=24.0)

    parser.add_argument("--curbside-conflict-distance-m", type=float, default=31.0)
    parser.add_argument("--curbside-target-forward-offset-m", type=float, default=-6.5)
    parser.add_argument("--curbside-target-start-lateral-offset-m", type=float, default=5.5)
    parser.add_argument("--curbside-target-end-lateral-offset-m", type=float, default=2.6)
    parser.add_argument("--curbside-occluder-lateral-offset-m", type=float, default=2.8)
    parser.add_argument("--curbside-occluder-count", type=int, default=1)
    parser.add_argument("--curbside-slot-1-forward-m", type=float, default=-7.5)
    parser.add_argument("--curbside-occluder-blueprint", default="vehicle.sprinter.mercedes")

    parser.add_argument("--sensor-x", type=float, default=1.8)
    parser.add_argument("--sensor-y", type=float, default=0.0)
    parser.add_argument("--sensor-z", type=float, default=1.55)
    parser.add_argument("--sensor-pitch", type=float, default=-4.0)
    parser.add_argument("--sensor-yaw", type=float, default=0.0)
    parser.add_argument("--sensor-roll", type=float, default=0.0)

    parser.add_argument("--lidar-range", type=float, default=120.0)
    parser.add_argument("--lidar-upper-fov", type=float, default=15.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-15.0)
    parser.add_argument("--lidar-channels", type=int, default=64)
    parser.add_argument("--lidar-rotation-frequency", type=float, default=20.0)
    parser.add_argument("--lidar-pps", type=int, default=600000)
    parser.add_argument("--lidar-sensor-tick", type=float, default=0.0)

    parser.add_argument("--gt-max-distance-m", type=float, default=140.0)
    parser.add_argument("--bbox-margin-xy", type=float, default=0.35)
    parser.add_argument("--bbox-margin-z-up", type=float, default=0.35)
    parser.add_argument("--bbox-margin-z-down", type=float, default=0.70)
    parser.add_argument(
        "--person-association-mode",
        choices=("bbox", "radius"),
        default="radius",
        help="Use radius for pedestrian geometry association; vehicles still use actor boxes.",
    )
    parser.add_argument("--person-association-radius-m", type=float, default=1.1)
    parser.add_argument("--person-association-z-down-m", type=float, default=0.4)
    parser.add_argument("--person-association-z-up-m", type=float, default=5.0)
    parser.add_argument("--min-vehicle-points", type=int, default=20)
    parser.add_argument("--min-person-points", type=int, default=2)
    parser.add_argument("--semantic-ped-tags", default="4,12,24,25")
    parser.add_argument("--semantic-vehicle-tags", default="10,14,15,16")
    parser.add_argument("--sample-points-per-frame", type=int, default=400)
    parser.add_argument("--debug-every", type=int, default=20)
    return parser.parse_args()


def experiment_id() -> str:
    return f"curbside_raw_vs_semantic_{time.strftime('%Y%m%d_%H%M%S')}"


def make_layout(args: argparse.Namespace, world: "carla.World", client: "carla.Client", tm: "carla.TrafficManager"):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No CARLA spawn points available.")
    anchor_index = int(args.anchor_spawn_index) % len(spawn_points)
    anchor = spawn_points[anchor_index].location
    spec = harness.SCENARIOS["curbside_parked_vehicle_pedestrian_occlusion"]
    rng = random.Random(int(args.seed))
    return harness.spawn_curbside_parked_pedestrian_layout(
        world,
        client,
        tm,
        spawn_points,
        anchor,
        spec,
        rng,
        ego_autopilot=False,
        route_choice="straight",
        ego_spawn_index=int(args.ego_spawn_index),
        curbside_conflict_distance_m=float(args.curbside_conflict_distance_m),
        curbside_occluder_lateral_offset_m=float(args.curbside_occluder_lateral_offset_m),
        curbside_target_start_lateral_offset_m=float(args.curbside_target_start_lateral_offset_m),
        curbside_target_end_lateral_offset_m=float(args.curbside_target_end_lateral_offset_m),
        curbside_target_forward_offset_m=float(args.curbside_target_forward_offset_m),
        curbside_ego_start_forward_m=0.0,
        helper_vehicle=False,
        helper_drive=False,
        slot_1_forward_m=float(args.curbside_slot_1_forward_m),
        occluder_count=int(args.curbside_occluder_count),
        forced_occluder_blueprint_id=str(args.curbside_occluder_blueprint or ""),
        occluder_z_offset_m=0.0,
    )


def make_event_monitor(
    args: argparse.Namespace,
    world: "carla.World",
    ego: "carla.Actor",
    layout: Dict[str, object],
) -> "harness.OcclusionEventMonitor":
    target_actor_id = layout.get("target_actor_id")
    target_actor = world.get_actor(int(target_actor_id)) if target_actor_id is not None else None
    target_end = harness.location_from_dict(layout["target_crossing_end_location"])
    target_trigger = harness.location_from_dict(layout["target_crossing_trigger_location"])
    route_rows = layout.get("controller_route_transforms") or []
    route_transforms = [
        harness.transform_from_dict(row)
        for row in route_rows
        if isinstance(row, dict) and "location" in row and "rotation" in row
    ]
    layout_control_speed = layout.get("target_crossing_control_speed_override")
    control_speed = (
        float(args.target_crossing_control_speed)
        if float(args.target_crossing_control_speed) > 0.0
        else None if layout_control_speed is None else float(layout_control_speed)
    )
    scripted_ego_drive = str(args.ego_motion) == "drive"
    route_lead_m = (
        float(args.target_crossing_trigger_route_lead_m)
        if scripted_ego_drive
        else 0.0
    )
    monitor = harness.OcclusionEventMonitor(
        world,
        ego,
        target_actor,
        target_end,
        scripted_ego_drive=scripted_ego_drive,
        ego_drive_mode="waypoint",
        ego_route_choice="straight",
        ego_route_transforms=route_transforms,
        ego_drive_throttle=float(args.ego_drive_throttle),
        ego_target_speed=float(args.ego_target_speed),
        ego_route_lookahead=float(args.ego_route_lookahead_m),
        target_crossing=True,
        target_crossing_delay_s=float(args.target_crossing_delay_s),
        target_crossing_speed=float(args.target_crossing_speed),
        target_crossing_trigger_location=target_trigger,
        target_crossing_trigger_distance_m=0.0,
        target_crossing_trigger_ttc_s=0.0,
        target_crossing_trigger_route_lead_m=route_lead_m,
        target_motion_mode="walker_control",
        target_crossing_control_speed_override=control_speed,
    )
    monitor.spawn()
    return monitor


def stop_sensor(sensor: Optional["carla.Actor"]) -> None:
    if sensor is not None:
        try:
            sensor.stop()
        except RuntimeError:
            pass


def main() -> int:
    args = parse_args()
    if carla is None:
        raise SystemExit("Could not import carla. Run inside the CARLA PythonAPI environment.")
    harness.carla = carla

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.experiment_id or experiment_id()
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout_s))
    world = client.load_world(args.town) if args.load_town else client.get_world()
    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_random_device_seed(int(args.seed))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    original_settings = world.get_settings()
    actors_to_destroy: List["carla.Actor"] = []
    raw_sensor = None
    semantic_sensor = None
    camera_sensor = None
    event_monitor = None
    raw_queue: "queue.Queue[carla.LidarMeasurement]" = queue.Queue()
    semantic_queue: "queue.Queue[carla.SemanticLidarMeasurement]" = queue.Queue()
    camera_queue: "queue.Queue[carla.Image]" = queue.Queue()

    try:
        if not args.asynch:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / float(args.fps)
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.tick()

        ego, special_actors, layout = make_layout(args, world, client, traffic_manager)
        actors_to_destroy.extend([ego, *special_actors])
        event_monitor = make_event_monitor(args, world, ego, layout)

        target_actor_id = int(layout["target_actor_id"])
        occluder_ids = [int(value) for value in layout.get("occluder_actor_ids", [])]
        print(
            "Curbside LiDAR diagnostic: "
            f"ego={ego.id} ego_motion={args.ego_motion} "
            f"target={target_actor_id} occluders={occluder_ids}"
        )

        sensor_tf = carla.Transform(
            carla.Location(x=args.sensor_x, y=args.sensor_y, z=args.sensor_z),
            carla.Rotation(pitch=args.sensor_pitch, yaw=args.sensor_yaw, roll=args.sensor_roll),
        )
        bp_lib = world.get_blueprint_library()
        sensor_tick = float(args.lidar_sensor_tick) if not args.asynch else 0.0
        if args.preview:
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(args.camera_width))
            camera_bp.set_attribute("image_size_y", str(args.camera_height))
            camera_bp.set_attribute("fov", str(args.camera_fov))
            camera_bp.set_attribute("sensor_tick", str(sensor_tick))
            camera_sensor = world.spawn_actor(camera_bp, sensor_tf, attach_to=ego)
            actors_to_destroy.append(camera_sensor)
            camera_sensor.listen(camera_queue.put)

        raw_bp = bp_lib.find("sensor.lidar.ray_cast")
        semantic_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
        lidar_diag.configure_lidar_bp(raw_bp, args, sensor_tick)
        lidar_diag.configure_lidar_bp(semantic_bp, args, sensor_tick)

        raw_sensor = world.spawn_actor(raw_bp, sensor_tf, attach_to=ego)
        semantic_sensor = world.spawn_actor(semantic_bp, sensor_tf, attach_to=ego)
        actors_to_destroy.extend([raw_sensor, semantic_sensor])
        raw_sensor.listen(raw_queue.put)
        semantic_sensor.listen(semantic_queue.put)

        for _ in range(max(0, int(args.warmup_ticks))):
            event_monitor.tick()
            world.tick() if not args.asynch else time.sleep(1.0 / float(args.fps))
            while not raw_queue.empty():
                raw_queue.get_nowait()
            while not semantic_queue.empty():
                semantic_queue.get_nowait()
            while not camera_queue.empty():
                camera_queue.get_nowait()

        frame_rows: List[dict] = []
        actor_rows: List[dict] = []
        raw_sample_rows: List[dict] = []
        semantic_sample_rows: List[dict] = []
        ped_tags = lidar_diag.parse_int_set(args.semantic_ped_tags)
        veh_tags = lidar_diag.parse_int_set(args.semantic_vehicle_tags)

        target_frames = int(round(float(args.duration_s) * float(args.fps)))
        start_wall = time.time()
        captured = 0

        for _ in range(target_frames):
            event_monitor.tick()
            frame_id = int(world.tick()) if not args.asynch else -1
            if args.asynch:
                time.sleep(1.0 / float(args.fps))

            try:
                raw_data = raw_queue.get(timeout=2.0)
                sem_data = semantic_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            while not raw_queue.empty():
                raw_data = raw_queue.get_nowait()
            while not semantic_queue.empty():
                sem_data = semantic_queue.get_nowait()

            camera_data = None
            if args.preview:
                try:
                    camera_data = camera_queue.get_nowait()
                    while not camera_queue.empty():
                        camera_data = camera_queue.get_nowait()
                except queue.Empty:
                    camera_data = None

            raw_arr = lidar_diag.raw_lidar_to_array(raw_data)
            sem_points_sensor, sem_tags, sem_obj_ids = lidar_diag.semantic_lidar_to_arrays(sem_data)
            raw_points_sensor = raw_arr[:, :3] if raw_arr.size else np.empty((0, 3), dtype=np.float32)
            raw_points_world = lidar_diag.transform_points(raw_points_sensor, raw_sensor.get_transform())
            sem_points_world = lidar_diag.transform_points(sem_points_sensor, semantic_sensor.get_transform())

            actor_boxes = lidar_diag.get_actor_boxes(world, ego, args.gt_max_distance_m)
            frame_actor_rows: List[dict] = []
            frame_actor_rows.extend(lidar_diag.evaluate_mode("raw_bbox", raw_points_world, actor_boxes, args))
            frame_actor_rows.extend(
                lidar_diag.evaluate_mode(
                    "semantic_tag_bbox",
                    sem_points_world,
                    actor_boxes,
                    args,
                    tags=sem_tags,
                    ped_tags=ped_tags,
                    veh_tags=veh_tags,
                )
            )
            frame_actor_rows.extend(
                lidar_diag.evaluate_mode(
                    "semantic_object_id",
                    sem_points_world,
                    actor_boxes,
                    args,
                    obj_ids=sem_obj_ids,
                )
            )
            for row in frame_actor_rows:
                row["frame"] = int(raw_data.frame)
                row["semantic_frame"] = int(sem_data.frame)
                row["is_target_actor"] = int(int(row["actor_id"]) == target_actor_id)
                actor_rows.append(row)

            mode_counts = lidar_diag.summarize_actor_rows(frame_actor_rows)
            target_rows = [row for row in frame_actor_rows if int(row["actor_id"]) == target_actor_id]
            target_counts = lidar_diag.summarize_actor_rows(target_rows)
            tag_counts = {
                str(int(tag)): int(count)
                for tag, count in zip(*np.unique(sem_tags, return_counts=True))
            } if sem_tags.size else {}

            raw_bytes = int(raw_points_sensor.shape[0] * 4 * 4)
            sem_bytes_est = int(sem_points_sensor.shape[0] * (3 * 4 + 2 * 4))
            frame_row = {
                "frame": int(raw_data.frame),
                "world_tick_frame": frame_id,
                "semantic_frame": int(sem_data.frame),
                "elapsed_wall_s": round(time.time() - start_wall, 4),
                "raw_points": int(raw_points_sensor.shape[0]),
                "semantic_points": int(sem_points_sensor.shape[0]),
                "raw_bytes_est": raw_bytes,
                "semantic_bytes_est": sem_bytes_est,
                "actor_vehicle_count": sum(1 for box in actor_boxes if box.actor_type == "vehicle"),
                "actor_person_count": sum(1 for box in actor_boxes if box.actor_type == "person"),
                "target_actor_id": target_actor_id,
                "target_started": int(bool(event_monitor.target_started)),
                "target_crossing_completed": int(bool(event_monitor.target_crossing_completed)),
                "raw_vehicle_recall": mode_counts.get("raw_bbox", {}).get("vehicle", {}).get("recall"),
                "raw_person_recall": mode_counts.get("raw_bbox", {}).get("person", {}).get("recall"),
                "semantic_tag_vehicle_recall": mode_counts.get("semantic_tag_bbox", {}).get("vehicle", {}).get("recall"),
                "semantic_tag_person_recall": mode_counts.get("semantic_tag_bbox", {}).get("person", {}).get("recall"),
                "semantic_id_vehicle_recall": mode_counts.get("semantic_object_id", {}).get("vehicle", {}).get("recall"),
                "semantic_id_person_recall": mode_counts.get("semantic_object_id", {}).get("person", {}).get("recall"),
                "target_raw_hit": target_counts.get("raw_bbox", {}).get("person", {}).get("recall"),
                "target_semantic_tag_hit": target_counts.get("semantic_tag_bbox", {}).get("person", {}).get("recall"),
                "target_semantic_id_hit": target_counts.get("semantic_object_id", {}).get("person", {}).get("recall"),
                "semantic_tag_counts_json": json.dumps(tag_counts, sort_keys=True),
            }
            frame_rows.append(frame_row)

            if args.preview and camera_data is not None:
                try:
                    import cv2

                    actor_markers = lidar_diag.project_actor_markers(
                        actor_boxes,
                        camera_sensor.get_transform(),
                        int(camera_data.width),
                        int(camera_data.height),
                        args.camera_fov,
                    )
                    preview = lidar_diag.draw_preview(
                        lidar_diag.camera_image_to_bgr(camera_data),
                        frame_row,
                        spawned_walkers=1,
                        spawned_vehicles=len(occluder_ids),
                        preview_width=args.preview_width,
                        preview_height=args.preview_height,
                        actor_markers=actor_markers,
                    )
                    cv2.imshow("SceneSense curbside raw-vs-semantic LiDAR diagnostic", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Preview requested stop via q.")
                        break
                except RuntimeError as exc:
                    print(f"Preview disabled: {exc}")
                    args.preview = False

            if args.sample_points_per_frame > 0:
                raw_n = min(args.sample_points_per_frame, raw_points_world.shape[0])
                if raw_n:
                    idx = np.random.choice(raw_points_world.shape[0], raw_n, replace=False)
                    for p in raw_points_world[idx]:
                        raw_sample_rows.append(
                            {"frame": int(raw_data.frame), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                        )
                sem_n = min(args.sample_points_per_frame, sem_points_world.shape[0])
                if sem_n:
                    idx = np.random.choice(sem_points_world.shape[0], sem_n, replace=False)
                    for p, tag, obj_id in zip(sem_points_world[idx], sem_tags[idx], sem_obj_ids[idx]):
                        semantic_sample_rows.append(
                            {
                                "frame": int(sem_data.frame),
                                "x": float(p[0]),
                                "y": float(p[1]),
                                "z": float(p[2]),
                                "tag": int(tag),
                                "tag_name": lidar_diag.CITYSCAPES_TAGS.get(int(tag), "unknown"),
                                "object_id": int(obj_id),
                            }
                        )

            captured += 1
            if args.debug_every > 0 and captured % args.debug_every == 0:
                print(
                    f"captured={captured}/{target_frames} target_started={int(event_monitor.target_started)} "
                    f"raw_pts={raw_points_sensor.shape[0]} sem_pts={sem_points_sensor.shape[0]}"
                )

        lidar_diag.write_csv(
            output_dir / "frame_metrics.csv",
            frame_rows,
            [
                "frame",
                "world_tick_frame",
                "semantic_frame",
                "elapsed_wall_s",
                "raw_points",
                "semantic_points",
                "raw_bytes_est",
                "semantic_bytes_est",
                "actor_vehicle_count",
                "actor_person_count",
                "target_actor_id",
                "target_started",
                "target_crossing_completed",
                "raw_vehicle_recall",
                "raw_person_recall",
                "semantic_tag_vehicle_recall",
                "semantic_tag_person_recall",
                "semantic_id_vehicle_recall",
                "semantic_id_person_recall",
                "target_raw_hit",
                "target_semantic_tag_hit",
                "target_semantic_id_hit",
                "semantic_tag_counts_json",
            ],
        )
        lidar_diag.write_csv(
            output_dir / "actor_metrics.csv",
            actor_rows,
            [
                "frame",
                "semantic_frame",
                "mode",
                "actor_type",
                "actor_id",
                "is_target_actor",
                "blueprint_id",
                "point_count",
                "hit",
                "xy_error_m",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "actor_x",
                "actor_y",
                "actor_z",
            ],
        )
        if raw_sample_rows:
            lidar_diag.write_csv(output_dir / "raw_points_sample.csv", raw_sample_rows, ["frame", "x", "y", "z"])
        if semantic_sample_rows:
            lidar_diag.write_csv(
                output_dir / "semantic_points_sample.csv",
                semantic_sample_rows,
                ["frame", "x", "y", "z", "tag", "tag_name", "object_id"],
            )
        if event_monitor is not None:
            lidar_diag.write_csv(
                output_dir / "curbside_event_trace.csv",
                event_monitor.trace_rows,
                list(event_monitor.trace_rows[0].keys()) if event_monitor.trace_rows else [],
            )

        summary = {
            "experiment_id": run_id,
            "output_dir": str(output_dir.resolve()),
            "captured_frames": captured,
            "settings": vars(args),
            "layout": layout,
            "target_actor_id": target_actor_id,
            "occluder_actor_ids": occluder_ids,
            "sensor": {
                "raw": "sensor.lidar.ray_cast",
                "semantic": "sensor.lidar.ray_cast_semantic",
                "same_transform": True,
                "lidar_range_m": args.lidar_range,
                "channels": args.lidar_channels,
                "points_per_second": args.lidar_pps,
                "sensor_tick": sensor_tick,
            },
            "modes": {
                "raw_bbox": "Raw LiDAR xyz points assigned to CARLA actor boxes for evaluation only.",
                "semantic_tag_bbox": "Semantic LiDAR points filtered by semantic tag, then assigned to actor boxes.",
                "semantic_object_id": "Semantic LiDAR points grouped by CARLA object id; this is oracle association.",
            },
            "overall": lidar_diag.summarize_actor_rows(actor_rows),
            "target_only": lidar_diag.summarize_actor_rows(
                [row for row in actor_rows if int(row["actor_id"]) == target_actor_id]
            ),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        stop_sensor(raw_sensor)
        stop_sensor(semantic_sensor)
        stop_sensor(camera_sensor)
        if args.preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        if event_monitor is not None:
            try:
                if event_monitor.target_controller is not None:
                    event_monitor.target_controller.stop()
            except RuntimeError:
                pass
            try:
                if event_monitor.collision_sensor is not None and event_monitor.collision_sensor.is_alive:
                    event_monitor.collision_sensor.destroy()
            except RuntimeError:
                pass
        for actor in reversed(actors_to_destroy):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        if not args.asynch:
            try:
                traffic_manager.set_synchronous_mode(False)
                world.apply_settings(original_settings)
            except RuntimeError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

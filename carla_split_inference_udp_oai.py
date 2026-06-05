#!/usr/bin/env python3

"""
CARLA split-inference demo, OAI 5G variant of carla_split_inference_udp_demo.py.

Same Faster R-CNN split-inference pipeline as the original. The added
`--role` flag controls which half of the pipeline runs in this process:

    --role loopback   (default; backwards-compatible)
        Both halves run in this process, all UDP on the loopback interface.
        Use this to confirm the baseline still passes after the refactor.

    --role front      (UE side, talks to a remote back-half)
        Runs CARLA + camera + backbone + feature-send. Receives detections
        back from the remote back-half. No RPN/ROI heads in this process.
        Bind sockets to the UE tunnel IP (--bind-host 10.0.0.2) and address
        the back-half via --remote-host (the perception container IP, e.g.
        192.168.70.140).

    --role back       (server side, receives features from a remote front-half)
        Runs only the RPN+ROI heads. No CARLA, no GUI, no metrics CSV.
        Bind sockets to whatever interface the front-half addresses
        (--bind-host 0.0.0.0 inside the container; --remote-host is the
        UE source IP that the front-half is sending from).

Run commands:

    # Loopback baseline (one process):
    python3 carla_split_inference_udp_oai.py --camera-resolution 1080p

    # 5G: front-half on host, back-half in the perception container:
    # (host)
    python3 carla_split_inference_udp_oai.py --role front \\
        --bind-host 10.0.0.2 --remote-host 192.168.70.140
    # (in oai-perception-rx)
    python3 carla_split_inference_udp_oai.py --role back \\
        --bind-host 0.0.0.0 --remote-host 10.0.0.2

Press `q` or `Esc` to exit the front/loopback GUI window. Ctrl+C in the
back-half terminal stops the inference worker cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import queue
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_carla():
    try:
        import carla as imported_carla

        return imported_carla
    except ModuleNotFoundError:
        pass

    script_path = Path(__file__).resolve()
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    search_roots = []
    for depth in (3, 2, 1, 0):
        if len(script_path.parents) > depth:
            search_roots.append(script_path.parents[depth])

    site_packages_candidates: List[Path] = []
    for root in search_roots:
        site_packages_candidates.extend(root.glob(f"**/lib/{py_version}/site-packages"))

    for site_packages in site_packages_candidates:
        if not list(site_packages.glob("carla*.so")):
            continue
        sys.path.insert(0, str(site_packages))
        try:
            import carla as imported_carla

            return imported_carla
        except ModuleNotFoundError:
            sys.path.pop(0)

    raise ModuleNotFoundError(
        "Unable to import CARLA. Install the CARLA Python API in a Python "
        f"{py_version} environment, or add its site-packages directory to PYTHONPATH."
    )


# Deferred so --role back can run in environments without the CARLA Python
# API (e.g. the oai-perception-rx container). Populated by ensure_carla()
# before any code path that actually touches CARLA.
carla = None


def ensure_carla():
    global carla
    if carla is None:
        carla = _bootstrap_carla()
    return carla

from torchvision.models.detection import (  # noqa: E402
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.image_list import ImageList  # noqa: E402


HEADER_STRUCT = struct.Struct("!IHH")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_WEIGHTS = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
COCO_LABELS = DEFAULT_WEIGHTS.meta["categories"]
SAFE_HERO_BLUEPRINT_IDS = (
    "vehicle.lincoln.mkz",
    "vehicle.lincoln.mkz_2020",
    "vehicle.lincoln.mkz_2017",
    "vehicle.mercedes.coupe_2020",
    "vehicle.dodge.charger_2020",
    "vehicle.audi.a2",
    "vehicle.toyota.prius",
    "vehicle.nissan.micra",
)
CAMERA_RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
DEFAULT_METRICS_LOG_DIR = Path(__file__).resolve().parent / "metrics_logs"
METRICS_CSV_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "frame_id",
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "payload_bytes",
    "payload_bytes_uncompressed",
    "payload_kib",
    "payload_uncompressed_kib",
    "payload_chunks",
    "detections",
    "gt_od_available",
    "gt_object_count",
    "gt_vehicle_count",
    "gt_person_count",
    "pred_object_count",
    "pred_vehicle_count",
    "pred_person_count",
    "od_match_iou_threshold",
    "od_matched_count",
    "od_vehicle_matched_count",
    "od_person_matched_count",
    "od_recall",
    "od_precision",
    "od_vehicle_recall",
    "od_vehicle_precision",
    "od_person_recall",
    "od_person_precision",
    "od_mean_iou",
)
DEFAULT_METRICS_BATCH_SIZE = 60
DEFAULT_METRICS_FLUSH_INTERVAL = 1.0
DEFAULT_LIVE_PLOT_REFRESH_SECONDS = 0.25
DEFAULT_METRICS_WARMUP_FRAMES = 30
COCO_VEHICLE_NAMES = {"bicycle", "car", "motorcycle", "bus", "train", "truck"}
COCO_PERSON_NAMES = {"person"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA split inference over UDP.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--town",
        default="Town10HD_Opt",
        help="Town to load before spawning actors.",
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
        help="Traffic Manager port for autopilot vehicles.",
    )
    parser.add_argument(
        "--vehicle-blueprint",
        default="vehicle.lincoln.mkz_2017",
        help="Blueprint id for the hero vehicle.",
    )
    parser.add_argument(
        "--camera-resolution",
        choices=["custom", *CAMERA_RESOLUTION_PRESETS.keys()],
        default="custom",
        help="Preset camera resolution. Use custom to honor --camera-width/--camera-height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera width used when --camera-resolution custom.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=384,
        help="Camera height used when --camera-resolution custom.",
    )
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a camera frame before retrying or failing warmup.",
    )
    parser.add_argument(
        "--camera-warmup-ticks",
        type=int,
        default=8,
        help="Maximum synchronous ticks used to wait for the first camera frame.",
    )
    parser.add_argument("--camera-x", type=float, default=1.6)
    parser.add_argument("--camera-z", type=float, default=1.7)
    parser.add_argument(
        "--npc-vehicles",
        type=int,
        default=20,
        help="Number of background autopilot vehicles to spawn.",
    )
    parser.add_argument(
        "--npc-pedestrians",
        type=int,
        default=30,
        help="Number of background pedestrians to spawn.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.55,
        help="Minimum confidence score shown on screen.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
        help="Maximum number of detections rendered per frame.",
    )
    od_gt_group = parser.add_mutually_exclusive_group()
    od_gt_group.add_argument(
        "--enable-od-gt",
        dest="enable_od_gt",
        action="store_true",
        help=(
            "Project CARLA vehicle/person actors into the RGB camera and log "
            "first-pass OD precision/recall using 2D IoU matching."
        ),
    )
    od_gt_group.add_argument(
        "--disable-od-gt",
        dest="enable_od_gt",
        action="store_false",
        help="Skip CARLA actor GT projection for OD quality metrics.",
    )
    parser.set_defaults(enable_od_gt=False)
    parser.add_argument(
        "--od-gt-iou-threshold",
        type=float,
        default=0.5,
        help="2D box IoU threshold used for camera-only OD GT matching.",
    )
    parser.add_argument(
        "--od-gt-min-area-px",
        type=float,
        default=64.0,
        help="Drop projected GT boxes smaller than this image area.",
    )
    parser.add_argument(
        "--od-gt-max-distance-m",
        type=float,
        default=80.0,
        help="Drop actor GT farther than this from the RGB camera. 0 disables.",
    )
    parser.add_argument(
        "--camera-source-port",
        type=int,
        default=36000,
        help="Local UDP source port used by the camera-side sender.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=36001,
        help="Local UDP receive port for the remote inference side.",
    )
    parser.add_argument(
        "--remote-source-port",
        type=int,
        default=36002,
        help="Local UDP source port used by the remote side sender.",
    )
    parser.add_argument(
        "--camera-result-port",
        type=int,
        default=36003,
        help="Local UDP receive port used by the camera side for detections.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=60000,
        help="Maximum UDP datagram size, including the custom header.",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=0.35,
        help="Seconds to wait for a matching detection result.",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=0.25,
        help="Socket timeout used by the background UDP threads.",
    )
    parser.add_argument(
        "--front-device",
        default="auto",
        help="Device for the first half of the model, e.g. auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--back-device",
        default="auto",
        help="Device for the second half of the model, e.g. auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional local Faster R-CNN checkpoint path.",
    )
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
        help="Skip pretrained torchvision weights and use random weights instead.",
    )
    parser.add_argument(
        "--metrics-log-dir",
        default=str(DEFAULT_METRICS_LOG_DIR),
        help="Directory where the metrics CSV and offline plot will be saved.",
    )
    parser.add_argument(
        "--metrics-log-prefix",
        default="split_inference_metrics",
        help="Filename prefix used for the metrics CSV and offline plot.",
    )
    parser.set_defaults(live_plot=True, collect_metrics=True)
    live_plot_group = parser.add_mutually_exclusive_group()
    live_plot_group.add_argument(
        "--enable-live-plot",
        dest="live_plot",
        action="store_true",
        help="Enable the real-time matplotlib metrics plot window.",
    )
    live_plot_group.add_argument(
        "--disable-live-plot",
        dest="live_plot",
        action="store_false",
        help="Disable the real-time matplotlib metrics plot window.",
    )
    metrics_collection_group = parser.add_mutually_exclusive_group()
    metrics_collection_group.add_argument(
        "--enable-data-collection",
        "--enable-metrics-collection",
        dest="collect_metrics",
        action="store_true",
        help="Enable metrics CSV logging and offline plot generation.",
    )
    metrics_collection_group.add_argument(
        "--disable-data-collection",
        "--disable-metrics-collection",
        dest="collect_metrics",
        action="store_false",
        help="Disable metrics CSV logging and offline plot generation.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI windows and run without the OpenCV camera view or live plot.",
    )
    parser.add_argument(
        "--live-plot-history",
        type=int,
        default=300,
        help="Number of recent samples shown in the live metrics plot.",
    )
    parser.add_argument(
        "--live-plot-update-interval",
        type=int,
        default=10,
        help="Send one live-plot update every N processed frames.",
    )
    parser.add_argument(
        "--metrics-batch-size",
        type=int,
        default=DEFAULT_METRICS_BATCH_SIZE,
        help="Number of queued metrics samples written to CSV per batch flush.",
    )
    parser.add_argument(
        "--metrics-flush-interval",
        type=float,
        default=DEFAULT_METRICS_FLUSH_INTERVAL,
        help="Maximum seconds between background CSV flushes.",
    )
    parser.add_argument(
        "--metrics-queue-size",
        type=int,
        default=2048,
        help="Maximum number of queued metrics samples before old samples are dropped.",
    )
    parser.add_argument(
        "--metrics-warmup-frames",
        type=int,
        default=DEFAULT_METRICS_WARMUP_FRAMES,
        help=(
            "Number of initial frames excluded from metrics while per-scale feature "
            "range trackers stabilize."
        ),
    )
    parser.add_argument(
        "--live-plot-refresh-seconds",
        type=float,
        default=DEFAULT_LIVE_PLOT_REFRESH_SECONDS,
        help="Seconds between GUI refreshes inside the live plot worker.",
    )
    parser.add_argument(
        "--metrics-plot-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--role",
        choices=["loopback", "front", "back"],
        default="loopback",
        help=(
            "Which half of the split-inference pipeline this process runs. "
            "'loopback' = both halves in one process (default, preserves the "
            "original demo's behavior). 'front' = camera + backbone only. "
            "'back' = RPN+ROI heads only."
        ),
    )
    parser.add_argument(
        "--bind-host",
        default=DEFAULT_HOST,
        help=(
            "Local interface address used when binding UDP sockets. For the "
            "5G demo: --bind-host 10.0.0.2 on the UE side (front), "
            "--bind-host 0.0.0.0 in the perception container (back)."
        ),
    )
    parser.add_argument(
        "--remote-host",
        default=None,
        help=(
            "IP of the other half of the split. Defaults to --bind-host "
            "(loopback). For 5G: --remote-host 192.168.70.140 on front, "
            "--remote-host 10.0.0.2 on back."
        ),
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    return device


def resolve_camera_dimensions(args: argparse.Namespace) -> Tuple[int, int, str]:
    if args.camera_resolution != "custom":
        width, height = CAMERA_RESOLUTION_PRESETS[args.camera_resolution]
        return width, height, args.camera_resolution

    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("camera width and height must be positive integers.")

    return args.camera_width, args.camera_height, f"custom {args.camera_width}x{args.camera_height}"


def resolve_metrics_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_dir = Path(args.metrics_log_dir).expanduser().resolve()
    prefix = args.metrics_log_prefix.strip() or "split_inference_metrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    plot_path = output_dir / f"{prefix}_{timestamp}.png"
    return csv_path, plot_path


def has_graphical_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def choose_vehicle_blueprints(
    world: "carla.World",
    cars_only: bool = False,
) -> List["carla.ActorBlueprint"]:
    blueprints = []
    for blueprint in world.get_blueprint_library().filter("vehicle.*"):
        if blueprint.has_attribute("number_of_wheels"):
            try:
                if int(blueprint.get_attribute("number_of_wheels").as_int()) != 4:
                    continue
            except RuntimeError:
                pass
        if cars_only and blueprint.has_attribute("base_type"):
            if str(blueprint.get_attribute("base_type")) != "car":
                continue
        blueprints.append(blueprint)
    if not blueprints and cars_only:
        return choose_vehicle_blueprints(world, cars_only=False)
    return sorted(blueprints, key=lambda blueprint: blueprint.id)


def choose_pedestrian_blueprints(world: "carla.World") -> List["carla.ActorBlueprint"]:
    return sorted(
        world.get_blueprint_library().filter("walker.pedestrian.*"),
        key=lambda blueprint: blueprint.id,
    )


def configure_vehicle_blueprint(
    blueprint: "carla.ActorBlueprint",
    role_name: str,
) -> "carla.ActorBlueprint":
    configured = blueprint
    if configured.has_attribute("role_name"):
        configured.set_attribute("role_name", role_name)
    if configured.has_attribute("color"):
        values = configured.get_attribute("color").recommended_values
        if values:
            configured.set_attribute("color", random.choice(values))
    if configured.has_attribute("driver_id"):
        values = configured.get_attribute("driver_id").recommended_values
        if values:
            configured.set_attribute("driver_id", random.choice(values))
    return configured


def configure_pedestrian_blueprint(
    blueprint: "carla.ActorBlueprint",
) -> "carla.ActorBlueprint":
    configured = blueprint
    if configured.has_attribute("is_invincible"):
        configured.set_attribute("is_invincible", "false")
    return configured


def get_fresh_vehicle_blueprint(
    world: "carla.World",
    blueprint_id: str,
    role_name: str,
) -> "carla.ActorBlueprint":
    return configure_vehicle_blueprint(
        world.get_blueprint_library().find(blueprint_id),
        role_name,
    )


def get_fresh_pedestrian_blueprint(
    world: "carla.World",
    blueprint_id: str,
) -> "carla.ActorBlueprint":
    return configure_pedestrian_blueprint(
        world.get_blueprint_library().find(blueprint_id),
    )


def resolve_hero_blueprint(
    world: "carla.World",
    blueprint_id: str,
) -> Tuple["carla.ActorBlueprint", bool]:
    blueprints = choose_vehicle_blueprints(world, cars_only=False)
    by_id = {blueprint.id: blueprint for blueprint in blueprints}

    preferred = by_id.get(blueprint_id)
    if preferred is not None:
        return preferred, False

    for fallback_id in SAFE_HERO_BLUEPRINT_IDS:
        preferred = by_id.get(fallback_id)
        if preferred is not None:
            return preferred, True

    fallback_blueprints = choose_vehicle_blueprints(world, cars_only=True)
    if not fallback_blueprints:
        raise RuntimeError("No suitable vehicle blueprints were found in the current CARLA world.")
    return fallback_blueprints[0], True


def try_spawn_vehicle_with_autopilot(
    client: "carla.Client",
    world: "carla.World",
    blueprint_id: str,
    spawn_point: "carla.Transform",
    role_name: str,
    traffic_manager_port: int,
) -> Optional["carla.Vehicle"]:
    command = carla.command.SpawnActor(
        get_fresh_vehicle_blueprint(world, blueprint_id, role_name),
        spawn_point,
    ).then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager_port))
    response = client.apply_batch_sync([command], True)[0]
    if response.error:
        return None
    return world.get_actor(response.actor_id)


def spawn_hero_vehicle(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    blueprint_id: str,
) -> "carla.Vehicle":
    preferred, fell_back = resolve_hero_blueprint(world, blueprint_id)
    if fell_back:
        print(
            f"Requested blueprint {blueprint_id!r} was not found. "
            f"Falling back to {preferred.id!r}."
        )

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        actor = try_spawn_vehicle_with_autopilot(
            client,
            world,
            preferred.id,
            spawn_point,
            role_name="hero",
            traffic_manager_port=traffic_manager.get_port(),
        )
        if actor is not None:
            return actor

    raise RuntimeError("Unable to spawn the hero vehicle at any spawn point.")


def spawn_background_traffic(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    count: int,
    hero_vehicle: "carla.Vehicle",
) -> List["carla.Vehicle"]:
    if count <= 0:
        return []

    blueprint_ids = [blueprint.id for blueprint in choose_vehicle_blueprints(world, cars_only=True)]
    spawn_points = world.get_map().get_spawn_points()
    hero_location = hero_vehicle.get_location()

    random.shuffle(spawn_points)
    batch = []
    for spawn_point in spawn_points:
        if len(batch) >= count:
            break
        if spawn_point.location.distance(hero_location) < 8.0:
            continue

        command = carla.command.SpawnActor(
            get_fresh_vehicle_blueprint(
                world,
                random.choice(blueprint_ids),
                "autopilot",
            ),
            spawn_point,
        ).then(
            carla.command.SetAutopilot(
                carla.command.FutureActor,
                True,
                traffic_manager.get_port(),
            )
        )
        batch.append(command)

    spawned: List["carla.Vehicle"] = []
    if batch:
        responses = client.apply_batch_sync(batch, True)
        for response in responses:
            if response.error:
                continue
            actor = world.get_actor(response.actor_id)
            if actor is not None:
                spawned.append(actor)

    if len(spawned) < count:
        print(f"Spawned {len(spawned)} background vehicles instead of requested {count}.")
    return spawned


def resolve_pedestrian_speed(blueprint: "carla.ActorBlueprint") -> float:
    if blueprint.has_attribute("speed"):
        values = list(blueprint.get_attribute("speed").recommended_values)
        if len(values) >= 2:
            return float(values[1])
        if values:
            return float(values[-1])
    return 1.2


def choose_pedestrian_spawn_points(
    world: "carla.World",
    hero_location: "carla.Location",
    count: int,
) -> List["carla.Transform"]:
    spawn_points: List["carla.Transform"] = []
    attempts = max(count * 12, 24)
    for _ in range(attempts):
        if len(spawn_points) >= count:
            break
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        if location.distance(hero_location) < 10.0:
            continue
        spawn_points.append(
            carla.Transform(
                carla.Location(x=location.x, y=location.y, z=location.z + 1.0)
            )
        )
    return spawn_points


def spawn_background_pedestrians(
    client: "carla.Client",
    world: "carla.World",
    count: int,
    hero_vehicle: "carla.Vehicle",
) -> Tuple[List["carla.Actor"], List["carla.Actor"]]:
    if count <= 0:
        return [], []

    pedestrian_blueprints = choose_pedestrian_blueprints(world)
    if not pedestrian_blueprints:
        print("No pedestrian blueprints were found in the current CARLA world.")
        return [], []

    spawn_points = choose_pedestrian_spawn_points(
        world,
        hero_vehicle.get_location(),
        count,
    )
    if len(spawn_points) < count:
        print(
            f"Found {len(spawn_points)} pedestrian spawn locations instead of requested {count}."
        )

    walker_batch = []
    walker_speeds: List[float] = []
    for spawn_point in spawn_points:
        blueprint_id = random.choice(pedestrian_blueprints).id
        blueprint = get_fresh_pedestrian_blueprint(world, blueprint_id)
        walker_batch.append(carla.command.SpawnActor(blueprint, spawn_point))
        walker_speeds.append(resolve_pedestrian_speed(blueprint))

    walker_ids: List[int] = []
    spawned_walker_speeds: List[float] = []
    if walker_batch:
        responses = client.apply_batch_sync(walker_batch, True)
        for response, speed in zip(responses, walker_speeds):
            if response.error:
                continue
            walker_ids.append(response.actor_id)
            spawned_walker_speeds.append(speed)

    if len(walker_ids) < count:
        print(f"Spawned {len(walker_ids)} background pedestrians instead of requested {count}.")
    if not walker_ids:
        return [], []

    controller_blueprint = world.get_blueprint_library().find("controller.ai.walker")
    controller_batch = [
        carla.command.SpawnActor(controller_blueprint, carla.Transform(), walker_id)
        for walker_id in walker_ids
    ]
    controller_ids: List[int] = []
    controlled_speeds: List[float] = []
    responses = client.apply_batch_sync(controller_batch, True)
    for walker_id, speed, response in zip(walker_ids, spawned_walker_speeds, responses):
        if response.error:
            continue
        controller_ids.append(response.actor_id)
        controlled_speeds.append(speed)

    if len(controller_ids) < len(walker_ids):
        print(
            f"Initialized {len(controller_ids)} pedestrian controllers for {len(walker_ids)} walkers."
        )

    walker_actors = [world.get_actor(actor_id) for actor_id in walker_ids]
    walker_actors = [actor for actor in walker_actors if actor is not None]
    controller_pairs = []
    for controller_id, speed in zip(controller_ids, controlled_speeds):
        actor = world.get_actor(controller_id)
        if actor is not None:
            controller_pairs.append((actor, speed))
    controller_actors = [actor for actor, _ in controller_pairs]

    if controller_actors:
        world.set_pedestrians_cross_factor(1.0)
        world.tick()
        for controller, speed in controller_pairs:
            try:
                controller.start()
                destination = world.get_random_location_from_navigation()
                if destination is not None:
                    controller.go_to_location(destination)
                controller.set_max_speed(float(speed))
            except RuntimeError:
                continue

    return walker_actors, controller_actors


def build_detector_model(args: argparse.Namespace) -> torch.nn.Module:
    num_classes = len(COCO_LABELS)

    if args.weights_path:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
        state_dict = torch.load(args.weights_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("Warning: checkpoint keys did not exactly match the detector.")
            if incompatible.missing_keys:
                print(f"  Missing keys: {len(incompatible.missing_keys)}")
            if incompatible.unexpected_keys:
                print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
    elif args.disable_pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
    else:
        try:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=DEFAULT_WEIGHTS)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the pretrained torchvision detector. The first run "
                "downloads the COCO weights. Re-run with internet access, pass "
                "--weights-path, or use --disable-pretrained to test the pipeline."
            ) from exc

    model.eval()
    return model


def clone_detector_model(reference_model: torch.nn.Module) -> torch.nn.Module:
    cloned = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=len(COCO_LABELS),
    )
    cloned.load_state_dict(reference_model.state_dict())
    cloned.eval()
    return cloned


class UDPMessageSocket:
    def __init__(
        self,
        bind_port: int,
        remote_port: Optional[int],
        chunk_bytes: int,
        socket_timeout: float,
        host: str = DEFAULT_HOST,
        remote_host: Optional[str] = None,
    ) -> None:
        if chunk_bytes <= HEADER_STRUCT.size:
            raise ValueError("chunk_bytes must be larger than the custom header size.")

        # `host` is the local bind address. `remote_host` is the destination IP
        # for sends. They're independent so we can bind to e.g. 10.0.0.2 (UE
        # tunnel) and send to 192.168.70.140 (perception container). When
        # `remote_host` is None we fall back to `host`, preserving the original
        # loopback behavior of the demo.
        self.host = host
        effective_remote_host = remote_host if remote_host is not None else host
        self.remote = (effective_remote_host, remote_port) if remote_port is not None else None
        self.chunk_bytes = chunk_bytes
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.socket.bind((host, bind_port))
        self.socket.settimeout(socket_timeout)
        self._pending: Dict[int, Dict[str, object]] = {}
        self._next_message_id = 1

    def close(self) -> None:
        self.socket.close()

    def send(self, payload: object) -> Tuple[int, int]:
        if self.remote is None:
            raise RuntimeError("This UDP socket does not have a configured remote address.")

        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(raw, level=1)
        max_payload = self.chunk_bytes - HEADER_STRUCT.size
        total_chunks = max(1, math.ceil(len(compressed) / max_payload))

        if total_chunks > 0xFFFF:
            raise ValueError("Serialized payload requires too many UDP chunks.")

        message_id = self._next_message_id
        self._next_message_id = (self._next_message_id + 1) & 0xFFFFFFFF
        if self._next_message_id == 0:
            self._next_message_id = 1

        for chunk_index in range(total_chunks):
            start = chunk_index * max_payload
            stop = start + max_payload
            chunk = compressed[start:stop]
            packet = HEADER_STRUCT.pack(message_id, chunk_index, total_chunks) + chunk
            self.socket.sendto(packet, self.remote)

        return len(compressed), total_chunks

    def receive(self) -> Optional[object]:
        while True:
            try:
                packet, _ = self.socket.recvfrom(self.chunk_bytes)
            except socket.timeout:
                self._drop_stale_buffers()
                return None
            except OSError:
                return None

            if len(packet) < HEADER_STRUCT.size:
                continue

            message_id, chunk_index, total_chunks = HEADER_STRUCT.unpack(
                packet[: HEADER_STRUCT.size]
            )
            if total_chunks == 0 or chunk_index >= total_chunks:
                continue

            now = time.time()
            entry = self._pending.get(message_id)
            if entry is None or int(entry["total_chunks"]) != total_chunks:
                entry = {
                    "updated_at": now,
                    "total_chunks": total_chunks,
                    "chunks": [None] * total_chunks,
                    "received": 0,
                }
                self._pending[message_id] = entry

            chunks = entry["chunks"]
            if chunks[chunk_index] is None:
                chunks[chunk_index] = packet[HEADER_STRUCT.size :]
                entry["received"] = int(entry["received"]) + 1
            entry["updated_at"] = now

            if int(entry["received"]) == total_chunks:
                combined = b"".join(chunk for chunk in chunks if chunk is not None)
                del self._pending[message_id]
                return pickle.loads(zlib.decompress(combined))

            self._drop_stale_buffers(now)

    def _drop_stale_buffers(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        stale = [
            message_id
            for message_id, entry in self._pending.items()
            if now - float(entry["updated_at"]) > 2.0
        ]
        for message_id in stale:
            del self._pending[message_id]


class DetectionResultStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._results: Dict[int, Dict[str, object]] = {}

    def put(self, frame_id: int, payload: Dict[str, object]) -> None:
        with self._condition:
            self._results[frame_id] = payload
            if len(self._results) > 120:
                oldest = sorted(self._results)[:20]
                for key in oldest:
                    self._results.pop(key, None)
            self._condition.notify_all()

    def wait_for(self, frame_id: int, timeout: float) -> Optional[Dict[str, object]]:
        deadline = time.time() + timeout
        with self._condition:
            while True:
                result = self._results.pop(frame_id, None)
                if result is not None:
                    return result
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)


class MetricsCSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRICS_CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self.sample_count = 0

    def append(self, record: Dict[str, object]) -> None:
        self._writer.writerow(record)
        self.sample_count += 1

    def append_many(self, records: List[Dict[str, object]]) -> None:
        if not records:
            return
        self._writer.writerows(records)
        self.sample_count += len(records)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def start_live_plot_worker(args: argparse.Namespace) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--metrics-plot-worker",
        "--live-plot-history",
        str(args.live_plot_history),
        "--live-plot-update-interval",
        str(args.live_plot_update_interval),
        "--live-plot-refresh-seconds",
        str(args.live_plot_refresh_seconds),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )


class AsyncMetricsCollector(threading.Thread):
    def __init__(
        self,
        csv_path: Optional[Path],
        enable_live_plot: bool,
        gui_enabled: bool,
        args: argparse.Namespace,
    ) -> None:
        super().__init__(daemon=True)
        self.queue: "queue.Queue[Optional[Dict[str, object]]]" = queue.Queue(
            maxsize=max(32, int(args.metrics_queue_size))
        )
        self.csv_logger = MetricsCSVLogger(csv_path) if csv_path is not None else None
        self.live_plot_process: Optional[subprocess.Popen] = None
        self.live_plot_stdin = None
        self.plot_send_interval = max(1, int(args.live_plot_update_interval))
        self.csv_batch_size = max(1, int(args.metrics_batch_size))
        self.csv_flush_interval = max(0.1, float(args.metrics_flush_interval))
        self.warning: Optional[str] = None
        self._stopped = threading.Event()
        self._dropped_samples = 0

        if enable_live_plot:
            if gui_enabled:
                try:
                    self.live_plot_process = start_live_plot_worker(args)
                    self.live_plot_stdin = self.live_plot_process.stdin
                except Exception as exc:
                    self.warning = f"Live metrics plot disabled: unable to start worker ({exc})"
            else:
                self.warning = "Live metrics plot disabled: running without a graphical display."

    def submit(self, record: Dict[str, object]) -> None:
        if self._stopped.is_set():
            return
        try:
            self.queue.put_nowait(record)
            return
        except queue.Full:
            pass

        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(record)
            self._dropped_samples += 1
        except queue.Full:
            self._dropped_samples += 1

    def close(self) -> None:
        self._stopped.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.join(timeout=5.0)
        if self.csv_logger is not None:
            self.csv_logger.close()
        if self.live_plot_stdin is not None:
            try:
                self.live_plot_stdin.close()
            except Exception:
                pass
        if self.live_plot_process is not None:
            try:
                self.live_plot_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.live_plot_process.terminate()
                try:
                    self.live_plot_process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.live_plot_process.kill()

    def run(self) -> None:
        csv_batch: List[Dict[str, object]] = []
        last_flush = time.monotonic()
        live_plot_counter = 0

        while True:
            timeout = max(0.05, self.csv_flush_interval / 2.0)
            try:
                record = self.queue.get(timeout=timeout)
            except queue.Empty:
                record = None

            if record is None:
                if self._stopped.is_set():
                    break
            else:
                if self.csv_logger is not None:
                    csv_batch.append(record)

                if self.live_plot_stdin is not None:
                    live_plot_counter += 1
                    if live_plot_counter % self.plot_send_interval == 0:
                        try:
                            self.live_plot_stdin.write(json.dumps(record) + "\n")
                            self.live_plot_stdin.flush()
                        except Exception:
                            self.live_plot_stdin = None
                            if self.live_plot_process is not None:
                                self.live_plot_process = None

            if (
                self.csv_logger is not None
                and csv_batch
                and (
                    len(csv_batch) >= self.csv_batch_size
                    or time.monotonic() - last_flush >= self.csv_flush_interval
                    or self._stopped.is_set()
                )
            ):
                self.csv_logger.append_many(csv_batch)
                self.csv_logger.flush()
                csv_batch.clear()
                last_flush = time.monotonic()

        if self.csv_logger is not None and csv_batch:
            self.csv_logger.append_many(csv_batch)
            self.csv_logger.flush()

        if self._dropped_samples > 0:
            print(
                f"Warning: dropped {self._dropped_samples} metrics samples to preserve real-time responsiveness.",
                file=sys.stderr,
            )


def get_camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    intrinsics = np.identity(3, dtype=np.float32)
    intrinsics[0, 0] = intrinsics[1, 1] = focal
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0
    return intrinsics


def world_to_camera_points(points_world: np.ndarray, cam_inv_matrix: np.ndarray) -> np.ndarray:
    if points_world.size == 0:
        return points_world
    points = np.concatenate(
        [
            points_world.astype(np.float32),
            np.ones((len(points_world), 1), dtype=np.float32),
        ],
        axis=1,
    )
    return (cam_inv_matrix @ points.T).T[:, :3]


def project_camera_points_to_image(
    points_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    u = intrinsics[0, 2] + (y2 / x2) * intrinsics[0, 0]
    v = intrinsics[1, 2] - (z2 / x2) * intrinsics[1, 1]
    return u, v, x2


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
        actor_transform = actor.get_transform()
    except RuntimeError:
        return None
    actor_matrix = np.array(actor_transform.get_matrix(), dtype=np.float32)
    bbox_location = np.array(
        [bbox.location.x, bbox.location.y, bbox.location.z],
        dtype=np.float32,
    )
    corners_local = bbox_corner_offsets(bbox.extent) + bbox_location.reshape(1, 3)
    homogeneous = np.concatenate(
        [corners_local, np.ones((corners_local.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (actor_matrix @ homogeneous.T).T[:, :3]


def projected_actor_bbox_xyxy(
    actor: "carla.Actor",
    cam_inv_matrix: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> Optional[Dict[str, float]]:
    corners_world = actor_bbox_world_corners(actor)
    if corners_world is None or corners_world.size == 0:
        return None
    corners_cam = world_to_camera_points(corners_world, cam_inv_matrix)
    u, v, depths = project_camera_points_to_image(corners_cam, intrinsics)
    if u.size == 0:
        return None
    x1 = float(np.clip(np.min(u), 0.0, width))
    y1 = float(np.clip(np.min(v), 0.0, height))
    x2 = float(np.clip(np.max(u), 0.0, width))
    y2 = float(np.clip(np.max(v), 0.0, height))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return None
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area_px": area,
        "center_x": (x1 + x2) / 2.0,
        "center_y": (y1 + y2) / 2.0,
        "min_depth_m": float(np.min(depths)),
    }


def normalize_coco_detection_label(detection: Dict[str, object]) -> Optional[str]:
    name = str(detection.get("name", "")).strip().lower()
    if name in COCO_PERSON_NAMES:
        return "person"
    if name in COCO_VEHICLE_NAMES:
        return "vehicle"
    return None


def project_od_ground_truth_objects(
    world: "carla.World",
    camera_actor: Optional["carla.Actor"],
    hero_id: Optional[int],
    *,
    width: int,
    height: int,
    fov: float,
    max_distance_m: float,
    min_area_px: float,
) -> List[Dict[str, object]]:
    if camera_actor is None:
        return []
    try:
        camera_transform = camera_actor.get_transform()
        camera_location = camera_transform.location
        cam_inv_matrix = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)
    except RuntimeError:
        return []

    intrinsics = get_camera_intrinsics(int(width), int(height), float(fov))
    gt_objects: List[Dict[str, object]] = []
    max_distance = max(0.0, float(max_distance_m))
    min_area = max(0.0, float(min_area_px))
    actor_sets = (("vehicle", "vehicle.*"), ("person", "walker.pedestrian.*"))
    for label, pattern in actor_sets:
        for actor in world.get_actors().filter(pattern):
            if hero_id is not None and actor.id == hero_id:
                continue
            if actor.id == camera_actor.id:
                continue
            try:
                distance_m = float(actor.get_location().distance(camera_location))
            except RuntimeError:
                continue
            if max_distance > 0.0 and distance_m > max_distance:
                continue
            projection = projected_actor_bbox_xyxy(
                actor,
                cam_inv_matrix,
                intrinsics,
                int(width),
                int(height),
            )
            if projection is None or float(projection["area_px"]) < min_area:
                continue
            gt_objects.append(
                {
                    "label": label,
                    "actor_id": int(actor.id),
                    "type_id": str(actor.type_id),
                    "bbox_xyxy": (
                        float(projection["x1"]),
                        float(projection["y1"]),
                        float(projection["x2"]),
                        float(projection["y2"]),
                    ),
                    "area_px": float(projection["area_px"]),
                    "center_x": float(projection["center_x"]),
                    "center_y": float(projection["center_y"]),
                    "distance_m": distance_m,
                    "depth_m": float(projection["min_depth_m"]),
                }
            )
    return gt_objects


def detection_bbox_xyxy(detection: Dict[str, object]) -> Optional[Tuple[float, float, float, float]]:
    raw = detection.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_iou_xyxy(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def empty_od_quality_columns(*, available: bool, iou_threshold: float) -> Dict[str, object]:
    return {
        "gt_od_available": int(available),
        "gt_object_count": 0,
        "gt_vehicle_count": 0,
        "gt_person_count": 0,
        "pred_object_count": 0,
        "pred_vehicle_count": 0,
        "pred_person_count": 0,
        "od_match_iou_threshold": float(iou_threshold),
        "od_matched_count": 0,
        "od_vehicle_matched_count": 0,
        "od_person_matched_count": 0,
        "od_recall": float("nan"),
        "od_precision": float("nan"),
        "od_vehicle_recall": float("nan"),
        "od_vehicle_precision": float("nan"),
        "od_person_recall": float("nan"),
        "od_person_precision": float("nan"),
        "od_mean_iou": float("nan"),
    }


def ratio_or_nan(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def compute_od_quality_columns(
    detections: Sequence[Dict[str, object]],
    gt_objects: Optional[Sequence[Dict[str, object]]],
    *,
    iou_threshold: float,
) -> Dict[str, object]:
    if gt_objects is None:
        return empty_od_quality_columns(available=False, iou_threshold=iou_threshold)

    predicted: List[Dict[str, object]] = []
    for detection in detections:
        label = normalize_coco_detection_label(detection)
        bbox = detection_bbox_xyxy(detection)
        if label is None or bbox is None:
            continue
        predicted.append({"label": label, "bbox_xyxy": bbox})

    gt_by_label = {
        "vehicle": [gt for gt in gt_objects if str(gt.get("label")) == "vehicle"],
        "person": [gt for gt in gt_objects if str(gt.get("label")) == "person"],
    }
    pred_by_label = {
        "vehicle": [pred for pred in predicted if str(pred.get("label")) == "vehicle"],
        "person": [pred for pred in predicted if str(pred.get("label")) == "person"],
    }

    candidates: List[Tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        pred_label = str(pred["label"])
        for gt_index, gt in enumerate(gt_objects):
            if pred_label != str(gt.get("label")):
                continue
            iou = bbox_iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])  # type: ignore[arg-type]
            if iou >= float(iou_threshold):
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(reverse=True)

    used_pred = set()
    used_gt = set()
    matched_ious: List[float] = []
    matched_by_label = {"vehicle": 0, "person": 0}
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matched_ious.append(float(iou))
        label = str(gt_objects[gt_index].get("label"))
        if label in matched_by_label:
            matched_by_label[label] += 1

    gt_count = len(gt_objects)
    pred_count = len(predicted)
    matched_count = len(matched_ious)
    return {
        "gt_od_available": 1,
        "gt_object_count": int(gt_count),
        "gt_vehicle_count": int(len(gt_by_label["vehicle"])),
        "gt_person_count": int(len(gt_by_label["person"])),
        "pred_object_count": int(pred_count),
        "pred_vehicle_count": int(len(pred_by_label["vehicle"])),
        "pred_person_count": int(len(pred_by_label["person"])),
        "od_match_iou_threshold": float(iou_threshold),
        "od_matched_count": int(matched_count),
        "od_vehicle_matched_count": int(matched_by_label["vehicle"]),
        "od_person_matched_count": int(matched_by_label["person"]),
        "od_recall": ratio_or_nan(matched_count, gt_count),
        "od_precision": ratio_or_nan(matched_count, pred_count),
        "od_vehicle_recall": ratio_or_nan(matched_by_label["vehicle"], len(gt_by_label["vehicle"])),
        "od_vehicle_precision": ratio_or_nan(matched_by_label["vehicle"], len(pred_by_label["vehicle"])),
        "od_person_recall": ratio_or_nan(matched_by_label["person"], len(gt_by_label["person"])),
        "od_person_precision": ratio_or_nan(matched_by_label["person"], len(pred_by_label["person"])),
        "od_mean_iou": float(sum(matched_ious) / len(matched_ious)) if matched_ious else float("nan"),
    }


def build_metrics_record(
    frame_id: int,
    elapsed_s: float,
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    detections: Sequence[Dict[str, object]],
    gt_objects: Optional[Sequence[Dict[str, object]]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    back_ms = float("nan")
    round_trip_ms = float("nan")
    if remote_stats is not None:
        back_ms = float(remote_stats["server_ms"])
        round_trip_ms = float(remote_stats["round_trip_ms"])

    payload_bytes = int(front_stats["payload_bytes"])
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    return {
        "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_s": float(elapsed_s),
        "frame_id": int(frame_id),
        "front_ms": float(front_stats["front_ms"]),
        "back_ms": back_ms,
        "round_trip_ms": round_trip_ms,
        "payload_bytes": payload_bytes,
        "payload_bytes_uncompressed": payload_bytes_uncompressed,
        "payload_kib": payload_bytes / 1024.0,
        "payload_uncompressed_kib": payload_bytes_uncompressed / 1024.0,
        "payload_chunks": int(front_stats["payload_chunks"]),
        "detections": int(len(detections)),
        **compute_od_quality_columns(
            detections,
            gt_objects,
            iou_threshold=float(args.od_gt_iou_threshold),
        ),
    }


def render_metrics_axes(
    axes: Tuple[object, object, object],
    records: List[Dict[str, object]],
    title: str,
) -> None:
    latency_ax, payload_ax, detections_ax = axes
    for axis in axes:
        axis.clear()
        axis.grid(True, alpha=0.3)

    latency_ax.set_title(title)
    latency_ax.set_ylabel("Latency (ms)")
    payload_ax.set_ylabel("Payload (KiB)")
    detections_ax.set_ylabel("Detections")
    detections_ax.set_xlabel("Elapsed time (s)")

    if not records:
        return

    elapsed = [float(record["elapsed_s"]) for record in records]
    front_ms = [float(record["front_ms"]) for record in records]
    back_ms = [float(record["back_ms"]) for record in records]
    round_trip_ms = [float(record["round_trip_ms"]) for record in records]
    payload_kib = [float(record["payload_kib"]) for record in records]
    payload_uncompressed_kib = [
        float(record.get("payload_uncompressed_kib", record["payload_kib"])) for record in records
    ]
    detections = [float(record["detections"]) for record in records]

    latency_ax.plot(elapsed, front_ms, label="Front half", color="tab:blue", linewidth=1.8)
    latency_ax.plot(elapsed, back_ms, label="Back half", color="tab:orange", linewidth=1.8)
    latency_ax.plot(
        elapsed,
        round_trip_ms,
        label="Round trip",
        color="tab:red",
        linewidth=1.8,
    )
    latency_ax.legend(loc="upper right")

    payload_ax.plot(
        elapsed,
        payload_kib,
        label="Compressed",
        color="tab:green",
        linewidth=1.8,
    )
    payload_ax.plot(
        elapsed,
        payload_uncompressed_kib,
        label="Float16 baseline",
        color="tab:gray",
        linewidth=1.4,
        linestyle="--",
    )
    payload_ax.legend(loc="upper right")
    detections_ax.plot(elapsed, detections, color="tab:purple", linewidth=1.8)


class LiveMetricsPlotter:
    def __init__(self, history_limit: int, update_interval: int, enable_window: bool) -> None:
        self.history_limit = max(10, history_limit)
        self.update_interval = max(1, update_interval)
        self.records: List[Dict[str, object]] = []
        self._updates = 0
        self.enabled = False
        self.warning: Optional[str] = None

        if not enable_window:
            self.warning = "Live metrics plot disabled: running without a graphical display."
            return

        try:
            import matplotlib

            matplotlib.use("TkAgg", force=True)
            import matplotlib.pyplot as plt

            self._plt = plt
            self._plt.ion()
            self._figure, axes = self._plt.subplots(
                3,
                1,
                figsize=(11, 8),
                sharex=True,
                constrained_layout=True,
            )
            self._axes = tuple(axes)
            try:
                self._figure.canvas.manager.set_window_title("CARLA Split Inference Metrics")
            except Exception:
                pass
            render_metrics_axes(
                self._axes,
                self.records,
                title="CARLA Split Inference Metrics (Live)",
            )
            self._figure.show()
            self.enabled = True
        except Exception as exc:
            self.warning = f"Live metrics plot disabled: unable to initialize TkAgg backend ({exc})"

    def update(self, record: Dict[str, object]) -> None:
        if not self.enabled:
            return
        if not self._plt.fignum_exists(self._figure.number):
            self.enabled = False
            return

        self.records.append(record)
        if len(self.records) > self.history_limit:
            self.records = self.records[-self.history_limit :]

        self._updates += 1
        if self._updates % self.update_interval != 0:
            return

        render_metrics_axes(
            self._axes,
            self.records,
            title="CARLA Split Inference Metrics (Live)",
        )
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)

    def pump_events(self) -> None:
        if not self.enabled:
            return
        if not self._plt.fignum_exists(self._figure.number):
            self.enabled = False
            return
        self._plt.pause(0.001)

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._plt.close(self._figure)
        except Exception:
            pass


def run_metrics_plot_worker(args: argparse.Namespace) -> int:
    import select

    plotter = LiveMetricsPlotter(
        history_limit=args.live_plot_history,
        update_interval=1,
        enable_window=True,
    )
    if plotter.warning:
        print(plotter.warning, file=sys.stderr)
    if not plotter.enabled:
        return 1

    try:
        while plotter.enabled:
            ready, _, _ = select.select([sys.stdin], [], [], args.live_plot_refresh_seconds)
            if ready:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                plotter.update(record)
            else:
                plotter.pump_events()
    finally:
        plotter.close()
    return 0


def load_metrics_records(csv_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row:
                continue
            records.append(
                {
                    "wall_time_iso": row["wall_time_iso"],
                    "elapsed_s": float(row["elapsed_s"]),
                    "frame_id": int(float(row["frame_id"])),
                    "front_ms": float(row["front_ms"]),
                    "back_ms": float(row["back_ms"]),
                    "round_trip_ms": float(row["round_trip_ms"]),
                    "payload_bytes": int(float(row["payload_bytes"])),
                    "payload_bytes_uncompressed": int(
                        float(row.get("payload_bytes_uncompressed", row["payload_bytes"]))
                    ),
                    "payload_kib": float(row["payload_kib"]),
                    "payload_uncompressed_kib": float(
                        row.get("payload_uncompressed_kib", row["payload_kib"])
                    ),
                    "payload_chunks": int(float(row["payload_chunks"])),
                    "detections": int(float(row["detections"])),
                }
            )
    return records


def generate_offline_metrics_plot(csv_path: Path, plot_path: Path) -> bool:
    records = load_metrics_records(csv_path)
    if not records:
        return False

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = tuple(figure.subplots(3, 1, sharex=True))
    render_metrics_axes(axes, records, title="CARLA Split Inference Metrics (Offline)")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    return True


TENSOR_INFO_STRUCT = struct.Struct("!IIIff")


@dataclass
class TensorInfo:
    shape: Tuple[int, int, int]
    min: float
    max: float

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError(f"TensorInfo expects a 3D shape, got {self.shape!r}")
        self.shape = tuple(int(value) for value in self.shape)
        self.min = float(self.min)
        self.max = float(self.max)

    def to_bytes(self) -> bytes:
        return TENSOR_INFO_STRUCT.pack(
            int(self.shape[0]),
            int(self.shape[1]),
            int(self.shape[2]),
            np.float32(self.min).item(),
            np.float32(self.max).item(),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TensorInfo":
        if len(data) != TENSOR_INFO_STRUCT.size:
            raise ValueError(
                f"TensorInfo payload must be {TENSOR_INFO_STRUCT.size} bytes, got {len(data)}"
            )
        channels, height, width, rmin, rmax = TENSOR_INFO_STRUCT.unpack(data)
        return cls(shape=(channels, height, width), min=rmin, max=rmax)


@dataclass
class FeatureFramePackOptions:
    quantize: bool = True
    bitdepth: int = 8
    tile: bool = True
    frame_div: int = 1


def compute_padding_2d(shape: Tuple[int, int], div: int) -> Tuple[int, int]:
    return (
        (shape[0] + div - 1) // div * div - shape[0],
        (shape[1] + div - 1) // div * div - shape[1],
    )


def compute_frame_resolution(shape: Tuple[int, int, int]) -> Tuple[int, int]:
    channels, height, width = shape

    short_edge = max(1, int(math.sqrt(channels)))
    while channels % short_edge != 0:
        short_edge -= 1
    long_edge = channels // short_edge

    height_edge = short_edge if height > width else long_edge
    width_edge = long_edge if height > width else short_edge
    return height_edge * height, width_edge * width


def compute_packed_frame_shape(
    tensor_shape: Tuple[int, int, int],
    batch_size: int,
    frame_div: int,
) -> Tuple[int, int, int]:
    frame_height, frame_width = compute_frame_resolution(tensor_shape)
    pad_h, pad_w = compute_padding_2d((frame_height, frame_width), frame_div)
    return int(batch_size), frame_height + pad_h, frame_width + pad_w


def tensor_to_tiled(x: torch.Tensor, tiled_frame_resolution: Tuple[int, int]) -> torch.Tensor:
    *head_dims, channels, height, width = x.shape
    frame_height, frame_width = tiled_frame_resolution
    channels_in_height = frame_height // height
    channels_in_width = frame_width // width
    if channels != channels_in_height * channels_in_width:
        raise ValueError(
            "Feature tensor channels do not match the computed tiled frame resolution."
        )

    return (
        x.reshape(*head_dims, channels_in_height, channels_in_width, height, width)
        .swapaxes(-3, -2)
        .reshape(*head_dims, frame_height, frame_width)
    )


def tiled_to_tensor(x_tiled: torch.Tensor, channel_resolution: Tuple[int, int]) -> torch.Tensor:
    height, width = channel_resolution
    *head_dims, frame_height, frame_width = x_tiled.shape
    channels_in_height = frame_height // height
    channels_in_width = frame_width // width
    channels = int(channels_in_height * channels_in_width)

    return (
        x_tiled.reshape(*head_dims, channels_in_height, height, channels_in_width, width)
        .swapaxes(-2, -3)
        .reshape(*head_dims, channels, height, width)
    )


def quantize(x: torch.Tensor, *, min: float, max: float, bitdepth: int) -> torch.Tensor:
    if bitdepth > 16:
        raise ValueError("Feature quantization bitdepth must be 16 or less.")

    span = float(max - min)
    if span <= 1e-12:
        return torch.zeros_like(x, dtype=torch.uint8 if bitdepth <= 8 else torch.uint16)

    max_level = (2**bitdepth) - 1
    x = ((x - min) / span).clip(0.0, 1.0)
    x = (x * max_level).round()
    return x.to(torch.uint8 if bitdepth <= 8 else torch.uint16)


def dequantize(x: torch.Tensor, *, min: float, max: float, bitdepth: int) -> torch.Tensor:
    max_level = (2**bitdepth) - 1
    x = x.to(torch.float32) / max_level
    return x * (max - min) + min


def symmetric_feature_channel_flipping(
    x: torch.Tensor,
    channel_resolution: Tuple[int, int],
) -> torch.Tensor:
    *head_dims, tiled_height, tiled_width = x.shape
    channel_height, channel_width = channel_resolution
    channels_in_height = tiled_height // channel_height
    channels_in_width = tiled_width // channel_width

    x = x.reshape(
        *head_dims,
        channels_in_height,
        channel_height,
        channels_in_width,
        channel_width,
    )
    x = x.clone()
    x[..., :, :, 1::2, :] = x[..., :, :, 1::2, :].flip(-1)
    x[..., 1::2, :, :, :] = x[..., 1::2, :, :, :].flip(-3)
    return x.reshape(*head_dims, tiled_height, tiled_width)


def inverse_symmetric_feature_channel_flipping(
    x: torch.Tensor,
    channel_resolution: Tuple[int, int],
) -> torch.Tensor:
    return symmetric_feature_channel_flipping(x, channel_resolution)


class FeatureFramePacker:
    def pack(
        self,
        features: torch.Tensor,
        tensor_info: TensorInfo,
        opts: FeatureFramePackOptions,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected a 4D feature tensor, got shape {tuple(features.shape)}")
        if features.dtype != torch.float32:
            raise ValueError(f"Expected float32 features, got {features.dtype}")
        if tuple(int(value) for value in features.shape[1:]) != tensor_info.shape:
            raise ValueError(
                "Feature tensor shape does not match tensor info: "
                f"{tuple(features.shape[1:])} vs {tensor_info.shape}"
            )

        x = features
        if opts.quantize:
            x = quantize(x, min=tensor_info.min, max=tensor_info.max, bitdepth=opts.bitdepth)

        if opts.tile:
            frame_resolution = compute_frame_resolution(tensor_info.shape)
            x = tensor_to_tiled(x, frame_resolution)
            x = symmetric_feature_channel_flipping(x, tensor_info.shape[-2:])
            pad_h, pad_w = compute_padding_2d(frame_resolution, opts.frame_div)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return x

    def unpack(
        self,
        arr: torch.Tensor,
        tensor_info: TensorInfo,
        opts: FeatureFramePackOptions,
    ) -> torch.Tensor:
        x = arr

        if opts.tile:
            frame_resolution = compute_frame_resolution(tensor_info.shape)
            pad_h, pad_w = compute_padding_2d(frame_resolution, opts.frame_div)
            if pad_h or pad_w:
                x = x[..., : frame_resolution[0], : frame_resolution[1]]
            x = inverse_symmetric_feature_channel_flipping(x, tensor_info.shape[-2:])
            x = tiled_to_tensor(x, tensor_info.shape[-2:])

        if opts.quantize:
            x = dequantize(x, min=tensor_info.min, max=tensor_info.max, bitdepth=opts.bitdepth)

        if tuple(int(value) for value in x.shape[1:]) != tensor_info.shape:
            raise ValueError(
                f"Decoded feature tensor shape {tuple(x.shape[1:])} does not match {tensor_info.shape}"
            )
        return x


class RangeTracker:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, current_min: float, current_max: float) -> Tuple[float, float]:
        alpha = self.alpha
        self._min = alpha * current_min + (1.0 - alpha) * min(self._min, current_min)
        self._max = alpha * current_max + (1.0 - alpha) * max(self._max, current_max)
        return self._min, self._max


@dataclass
class FeatureCodecPacket:
    feature_frame: np.ndarray
    tensor_info_bytes: bytes


class SimpleFeatureCodec:
    def __init__(self, device: torch.device, *, range_alpha: float = 0.1) -> None:
        self.device = device
        self.opts = FeatureFramePackOptions(
            quantize=True,
            bitdepth=8,
            tile=True,
            frame_div=2,
        )
        self.feature_frame_packer = FeatureFramePacker()
        self.range = RangeTracker(alpha=range_alpha)

    def encode(self, features: torch.Tensor) -> FeatureCodecPacket:
        features = features.detach().to(device=self.device, dtype=torch.float32)
        rmin, rmax = self.range.update(
            float(features.min().item()),
            float(features.max().item()),
        )
        tensor_info = TensorInfo(
            shape=tuple(int(value) for value in features.shape[1:]),
            min=rmin,
            max=rmax,
        )
        feature_frame_tensor = self.feature_frame_packer.pack(features, tensor_info, self.opts)
        feature_frame = np.ascontiguousarray(feature_frame_tensor.detach().to("cpu").numpy())
        return FeatureCodecPacket(
            feature_frame=feature_frame,
            tensor_info_bytes=tensor_info.to_bytes(),
        )

    def decode(
        self,
        feature_frame: np.ndarray,
        tensor_info_bytes: bytes,
    ) -> torch.Tensor:
        tensor_info = TensorInfo.from_bytes(tensor_info_bytes)
        feature_frame_tensor = torch.from_numpy(feature_frame).to(self.device)
        return self.feature_frame_packer.unpack(feature_frame_tensor, tensor_info, self.opts)


def _get_or_create_feature_codec(
    feature_codecs: Dict[str, SimpleFeatureCodec],
    name: str,
    device: torch.device,
) -> SimpleFeatureCodec:
    codec = feature_codecs.get(name)
    if codec is None:
        codec = SimpleFeatureCodec(device=device)
        feature_codecs[name] = codec
    return codec


def serialize_feature_maps(
    features: "OrderedDict[str, torch.Tensor]",
    feature_codecs: Dict[str, SimpleFeatureCodec],
) -> Tuple[Dict[str, Dict[str, bytes]], int]:
    serialized: Dict[str, Dict[str, bytes]] = {}
    payload_bytes_uncompressed = 0
    for name, tensor in features.items():
        codec = _get_or_create_feature_codec(feature_codecs, name, tensor.device)
        packet = codec.encode(tensor)
        payload_bytes_uncompressed += int(tensor.numel() * np.dtype(np.float16).itemsize)
        serialized[name] = {
            "feature_frame": packet.feature_frame.tobytes(),
            "tensor_info": packet.tensor_info_bytes,
        }
    return serialized, payload_bytes_uncompressed


def deserialize_feature_maps(
    serialized: Dict[str, Dict[str, bytes]],
    device: torch.device,
    batch_size: int,
    feature_codecs: Dict[str, SimpleFeatureCodec],
) -> "OrderedDict[str, torch.Tensor]":
    features: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, payload in serialized.items():
        codec = _get_or_create_feature_codec(feature_codecs, name, device)
        tensor_info = TensorInfo.from_bytes(payload["tensor_info"])
        frame_shape = compute_packed_frame_shape(
            tensor_info.shape,
            batch_size=batch_size,
            frame_div=codec.opts.frame_div,
        )
        feature_frame = (
            np.frombuffer(payload["feature_frame"], dtype=np.uint8).reshape(frame_shape).copy()
        )
        features[name] = codec.decode(feature_frame, payload["tensor_info"]).to(
            device=device,
            dtype=torch.float32,
        )
    return features


class CameraSideSplitInference:
    def __init__(
        self,
        model: torch.nn.Module,
        sender: UDPMessageSocket,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.sender = sender
        self.device = device
        self.feature_codecs: Dict[str, SimpleFeatureCodec] = OrderedDict()

    def process(self, frame_id: int, frame_bgr: np.ndarray) -> Dict[str, object]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=torch.float32)
            / 255.0
        )

        started = time.perf_counter()
        with torch.inference_mode():
            image_list, _ = self.model.transform([image_tensor], None)
            features = self.model.backbone(image_list.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

        serialized_features, payload_bytes_uncompressed = serialize_feature_maps(
            features,
            self.feature_codecs,
        )
        payload = {
            "frame_id": frame_id,
            "batch_shape": tuple(int(value) for value in image_list.tensors.shape),
            "image_sizes": [tuple(map(int, size)) for size in image_list.image_sizes],
            "original_sizes": [tuple(map(int, image_tensor.shape[-2:]))],
            "features": serialized_features,
            "camera_sent_perf": time.perf_counter(),
        }
        payload_bytes, payload_chunks = self.sender.send(payload)
        return {
            "front_ms": (time.perf_counter() - started) * 1000.0,
            "payload_bytes": payload_bytes,
            "payload_bytes_uncompressed": payload_bytes_uncompressed,
            "payload_chunks": payload_chunks,
        }


class RemoteInferenceWorker(threading.Thread):
    def __init__(
        self,
        model: torch.nn.Module,
        receiver: UDPMessageSocket,
        sender: UDPMessageSocket,
        device: torch.device,
        score_threshold: float,
        max_detections: int,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model.to(device).eval()
        self.receiver = receiver
        self.sender = sender
        self.device = device
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.stop_event = stop_event
        self.feature_codecs: Dict[str, SimpleFeatureCodec] = OrderedDict()

    def run(self) -> None:
        while not self.stop_event.is_set():
            payload = self.receiver.receive()
            if payload is None:
                continue

            try:
                detections = self._run_back_half(payload)
                self.sender.send(detections)
            except Exception as exc:
                print(f"Remote inference worker error: {exc}", file=sys.stderr)

    def _run_back_half(self, payload: Dict[str, object]) -> Dict[str, object]:
        started = time.perf_counter()
        batch_shape = tuple(int(value) for value in payload["batch_shape"])
        features = deserialize_feature_maps(
            payload["features"],
            self.device,
            batch_size=batch_shape[0],
            feature_codecs=self.feature_codecs,
        )
        image_sizes = [tuple(map(int, size)) for size in payload["image_sizes"]]
        original_sizes = [tuple(map(int, size)) for size in payload["original_sizes"]]

        dummy_images = torch.zeros(batch_shape, device=self.device)
        image_list = ImageList(dummy_images, image_sizes)

        with torch.inference_mode():
            proposals, _ = self.model.rpn(image_list, features, None)
            detections, _ = self.model.roi_heads(
                features,
                proposals,
                image_list.image_sizes,
                None,
            )
            detections = self.model.transform.postprocess(
                detections,
                image_list.image_sizes,
                original_sizes,
            )

        prediction = detections[0]
        scores = prediction["scores"].detach().cpu().numpy()
        keep = np.where(scores >= self.score_threshold)[0][: self.max_detections]

        serialized_detections: List[Dict[str, object]] = []
        boxes = prediction["boxes"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        for index in keep:
            label = int(labels[index])
            serialized_detections.append(
                {
                    "box": boxes[index].round(2).tolist(),
                    "score": float(scores[index]),
                    "label": label,
                    "name": COCO_LABELS[label] if label < len(COCO_LABELS) else str(label),
                }
            )

        return {
            "frame_id": int(payload["frame_id"]),
            "camera_sent_perf": float(payload["camera_sent_perf"]),
            "server_ms": (time.perf_counter() - started) * 1000.0,
            "detections": serialized_detections,
        }


class CameraResultReceiver(threading.Thread):
    def __init__(
        self,
        receiver: UDPMessageSocket,
        result_store: DetectionResultStore,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.receiver = receiver
        self.result_store = result_store
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            payload = self.receiver.receive()
            if payload is None:
                continue
            frame_id = int(payload["frame_id"])
            self.result_store.put(frame_id, payload)


def put_latest(q: "queue.Queue[carla.Image]", item: "carla.Image") -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def camera_image_to_bgr(image: "carla.Image") -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return np.ascontiguousarray(array[:, :, :3])


def wait_for_camera_frame(
    image_queue: "queue.Queue[carla.Image]",
    minimum_frame: int,
    timeout: float,
) -> Optional["carla.Image"]:
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        try:
            image = image_queue.get(timeout=remaining)
        except queue.Empty:
            return None
        if int(image.frame) < minimum_frame:
            continue
        return image


def warmup_camera_stream(
    world: "carla.World",
    image_queue: "queue.Queue[carla.Image]",
    warmup_ticks: int,
    timeout: float,
) -> "carla.Image":
    for _ in range(max(1, warmup_ticks)):
        world_frame = int(world.tick())
        image = wait_for_camera_frame(image_queue, world_frame, timeout)
        if image is not None:
            return image
    raise RuntimeError(
        "RGB camera did not produce any frames during startup. "
        "Try increasing --camera-timeout or --camera-warmup-ticks, and verify "
        "the CARLA server is rendering sensor data normally."
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    detections: List[Dict[str, object]],
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    metrics_warmup_remaining: int = 0,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(value) for value in det["box"]]
        label = int(det["label"])
        color = (
            int((37 * label) % 255),
            int((17 * label + 80) % 255),
            int((29 * label + 160) % 255),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{det['name']} {det['score']:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    payload_bytes = max(1, int(front_stats["payload_bytes"]))
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    compression_ratio = payload_bytes_uncompressed / payload_bytes
    lines = [
        f"Front half: {front_stats['front_ms']:.1f} ms",
        (
            "Feature payload: "
            f"{front_stats['payload_bytes'] / 1024.0:.1f} KiB "
            f"in {front_stats['payload_chunks']} UDP chunks"
        ),
        (
            "Float16 baseline: "
            f"{payload_bytes_uncompressed / 1024.0:.1f} KiB, ratio {compression_ratio:.2f}x"
        ),
        f"Detections: {len(detections)}",
    ]
    if remote_stats is not None:
        lines.append(f"Back half: {remote_stats['server_ms']:.1f} ms")
        lines.append(f"Round trip: {remote_stats['round_trip_ms']:.1f} ms")
    if metrics_warmup_remaining > 0:
        lines.append(f"Metrics warm-up: {metrics_warmup_remaining} frame(s) remaining")

    y = 24
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return annotated


def run_back_only(args: argparse.Namespace) -> None:
    """Back-half-only entry point (--role back).

    No CARLA, no GUI, no metrics CSV. Just builds the model, opens the two
    remote-side UDP sockets, runs RemoteInferenceWorker until Ctrl+C.
    """
    back_device = resolve_device(args.back_device)
    if back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base_model = build_detector_model(args)
    back_model = clone_detector_model(base_model)
    del base_model  # not used in back role

    remote_host = args.remote_host if args.remote_host is not None else args.bind_host

    remote_receiver = UDPMessageSocket(
        bind_port=args.remote_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
    )
    remote_sender = UDPMessageSocket(
        bind_port=args.remote_source_port,
        remote_port=args.camera_result_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
    )

    stop_event = threading.Event()
    remote_worker = RemoteInferenceWorker(
        model=back_model,
        receiver=remote_receiver,
        sender=remote_sender,
        device=back_device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        stop_event=stop_event,
    )
    remote_worker.start()

    print(
        f"[back] device={back_device} "
        f"recv {args.bind_host}:{args.remote_port}, "
        f"send -> {remote_host}:{args.camera_result_port}"
    )
    print("[back] Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[back] Interrupted.")
    finally:
        stop_event.set()
        try:
            remote_receiver.close()
        except OSError:
            pass
        try:
            remote_sender.close()
        except OSError:
            pass
        remote_worker.join(timeout=2.0)
        print("[back] Done.")


def run_demo(args: argparse.Namespace) -> None:
    # --role back short-circuits to the back-half-only path before any CARLA
    # or GUI imports run. --role front and the default --role loopback fall
    # through to the original demo body below, with the only differences
    # being:
    #   - UDP sockets bind/send to args.bind_host / args.remote_host
    #   - the in-process RemoteInferenceWorker is skipped in front role
    if args.role == "back":
        run_back_only(args)
        return

    # Front and loopback both drive CARLA; resolve the binding here so failures
    # surface before any sockets get opened.
    ensure_carla()

    random.seed(7)

    front_device = resolve_device(args.front_device)
    back_device = resolve_device(args.back_device)
    camera_width, camera_height, camera_resolution_label = resolve_camera_dimensions(args)
    metrics_csv_path: Optional[Path] = None
    metrics_plot_path: Optional[Path] = None
    if args.collect_metrics:
        metrics_csv_path, metrics_plot_path = resolve_metrics_output_paths(args)
    gui_enabled = has_graphical_display() and not args.headless

    if front_device.type == "cuda" or back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base_model = build_detector_model(args)
    # In --role front the back-half model isn't used in this process, but the
    # clone is cheap relative to the pretrained download and keeps the code
    # path uniform with loopback. Skipped device move keeps it on CPU.
    back_model = clone_detector_model(base_model) if args.role == "loopback" else None

    # Both halves use the same bind/remote hosts in loopback; in --role front
    # the four sockets use the UE source IP for bind and the back-half
    # container's IP for send.
    remote_host = args.remote_host if args.remote_host is not None else args.bind_host

    camera_sender = UDPMessageSocket(
        bind_port=args.camera_source_port,
        remote_port=args.remote_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
    )
    remote_receiver = UDPMessageSocket(
        bind_port=args.remote_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
    ) if args.role == "loopback" else None
    remote_sender = UDPMessageSocket(
        bind_port=args.remote_source_port,
        remote_port=args.camera_result_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
    ) if args.role == "loopback" else None
    camera_receiver = UDPMessageSocket(
        bind_port=args.camera_result_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
    )

    stop_event = threading.Event()
    result_store = DetectionResultStore()
    split_camera = CameraSideSplitInference(base_model, camera_sender, front_device)
    remote_worker = RemoteInferenceWorker(
        model=back_model,
        receiver=remote_receiver,
        sender=remote_sender,
        device=back_device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        stop_event=stop_event,
    ) if args.role == "loopback" else None
    result_receiver = CameraResultReceiver(
        receiver=camera_receiver,
        result_store=result_store,
        stop_event=stop_event,
    )
    if remote_worker is not None:
        remote_worker.start()
    result_receiver.start()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.load_world(args.town) if args.town else client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    world.tick()

    actors: List["carla.Actor"] = []
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)

    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"Town: {world.get_map().name}")
    print(f"Front device: {front_device}, back device: {back_device}")
    print(f"Camera resolution: {camera_width}x{camera_height} ({camera_resolution_label})")
    if args.collect_metrics and metrics_csv_path is not None and metrics_plot_path is not None:
        print(f"Metrics CSV: {metrics_csv_path}")
        print(f"Offline metrics plot: {metrics_plot_path}")
    else:
        print("Metrics data collection disabled. CSV logging and offline plot generation are off.")
    if not gui_enabled:
        if args.headless:
            print("GUI disabled by --headless. Running without the OpenCV view or live plot window.")
        else:
            print(
                "No graphical display detected. Running without the OpenCV view or live plot window."
            )
    print(
        "UDP ports: "
        f"camera {args.camera_source_port} -> remote {args.remote_port}, "
        f"remote {args.remote_source_port} -> camera {args.camera_result_port}"
    )

    metrics_collector = None
    if metrics_csv_path is not None or args.live_plot:
        metrics_collector = AsyncMetricsCollector(
            csv_path=metrics_csv_path,
            enable_live_plot=args.live_plot,
            gui_enabled=gui_enabled,
            args=args,
        )
        metrics_collector.start()
    if metrics_collector is not None and metrics_collector.warning:
        print(metrics_collector.warning)

    try:
        hero_vehicle = spawn_hero_vehicle(
            client,
            world,
            traffic_manager,
            args.vehicle_blueprint,
        )
        actors.append(hero_vehicle)
        print(f"Hero vehicle: {hero_vehicle.type_id}")

        background_vehicles = spawn_background_traffic(
            client,
            world,
            traffic_manager,
            args.npc_vehicles,
            hero_vehicle,
        )
        actors.extend(background_vehicles)
        if background_vehicles:
            print(f"Spawned {len(background_vehicles)} background vehicles.")

        pedestrian_walkers, pedestrian_controllers = spawn_background_pedestrians(
            client,
            world,
            args.npc_pedestrians,
            hero_vehicle,
        )
        actors.extend(pedestrian_walkers)
        actors.extend(pedestrian_controllers)
        if pedestrian_walkers:
            print(f"Spawned {len(pedestrian_walkers)} background pedestrians.")

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(camera_width))
        camera_bp.set_attribute("image_size_y", str(camera_height))
        camera_bp.set_attribute("fov", str(args.camera_fov))
        camera_bp.set_attribute("sensor_tick", str(1.0 / args.fps))
        camera_transform = carla.Transform(
            carla.Location(x=args.camera_x, z=args.camera_z)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=hero_vehicle)
        actors.append(camera)
        camera.listen(lambda image: put_latest(image_queue, image))
        first_image = warmup_camera_stream(
            world,
            image_queue,
            args.camera_warmup_ticks,
            args.camera_timeout,
        )
        print(f"Camera ready on frame {first_image.frame}.")
        metrics_start_perf: Optional[float] = None
        metrics_warmup_remaining = (
            max(0, int(args.metrics_warmup_frames)) if metrics_collector is not None else 0
        )
        if metrics_warmup_remaining == 0:
            metrics_start_perf = time.perf_counter()
        elif metrics_collector is not None:
            print(
                "Metrics warm-up: skipping the first "
                f"{metrics_warmup_remaining} frame(s) while feature range trackers stabilize."
            )

        if gui_enabled:
            cv2.namedWindow("CARLA Split Inference", cv2.WINDOW_AUTOSIZE)
        else:
            print("Headless run active. Press Ctrl+C to stop the demo.")

        while True:
            world_frame = int(world.tick())
            image = wait_for_camera_frame(image_queue, world_frame, args.camera_timeout)
            if image is None:
                print(
                    f"Warning: camera frame for world tick {world_frame} was not received "
                    f"within {args.camera_timeout:.1f}s; retrying."
                )
                continue
            frame_bgr = camera_image_to_bgr(image)
            front_stats = split_camera.process(image.frame, frame_bgr)

            result = result_store.wait_for(image.frame, args.result_timeout)
            remote_stats = None
            detections: List[Dict[str, object]] = []
            if result is not None:
                remote_stats = {
                    "server_ms": float(result["server_ms"]),
                    "round_trip_ms": (time.perf_counter() - float(result["camera_sent_perf"])) * 1000.0,
                }
                detections = list(result["detections"])

            gt_objects: Optional[List[Dict[str, object]]] = None
            if bool(args.enable_od_gt):
                gt_objects = project_od_ground_truth_objects(
                    world,
                    camera,
                    int(hero_vehicle.id),
                    width=int(camera_width),
                    height=int(camera_height),
                    fov=float(args.camera_fov),
                    max_distance_m=float(args.od_gt_max_distance_m),
                    min_area_px=float(args.od_gt_min_area_px),
                )

            if metrics_collector is not None:
                if metrics_warmup_remaining > 0:
                    metrics_warmup_remaining -= 1
                    if metrics_warmup_remaining == 0:
                        metrics_start_perf = time.perf_counter()
                else:
                    if metrics_start_perf is None:
                        metrics_start_perf = time.perf_counter()
                    metrics_record = build_metrics_record(
                        frame_id=image.frame,
                        elapsed_s=time.perf_counter() - metrics_start_perf,
                        front_stats=front_stats,
                        remote_stats=remote_stats,
                        detections=detections,
                        gt_objects=gt_objects,
                        args=args,
                    )
                    metrics_collector.submit(metrics_record)

            if gui_enabled:
                annotated = draw_overlay(
                    frame_bgr,
                    detections,
                    front_stats,
                    remote_stats,
                    metrics_warmup_remaining=metrics_warmup_remaining,
                )
                cv2.imshow("CARLA Split Inference", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        stop_event.set()

        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        for actor in reversed(actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except RuntimeError:
                pass
            try:
                actor.destroy()
            except RuntimeError:
                pass

        camera_sender.close()
        if remote_receiver is not None:
            remote_receiver.close()
        if remote_sender is not None:
            remote_sender.close()
        camera_receiver.close()

        if remote_worker is not None:
            remote_worker.join(timeout=1.0)
        result_receiver.join(timeout=1.0)
        if gui_enabled:
            cv2.destroyAllWindows()
        if metrics_collector is not None:
            metrics_collector.close()
        if metrics_csv_path is not None and metrics_plot_path is not None:
            try:
                if generate_offline_metrics_plot(metrics_csv_path, metrics_plot_path):
                    print(f"Saved offline metrics plot to {metrics_plot_path}")
                else:
                    print("No metrics samples were collected; skipping offline metrics plot.")
            except Exception as exc:
                print(f"Warning: unable to generate offline metrics plot: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.metrics_plot_worker:
        raise SystemExit(run_metrics_plot_worker(args))
    run_demo(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")

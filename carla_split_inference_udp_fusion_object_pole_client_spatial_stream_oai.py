#!/usr/bin/env python3

"""
Traffic-light-pole RGB+radar fusion client for split-inference object detection
over localhost or the OAI 5G transport path.

Sibling of carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py.
Hosts the trained pole_lraspp_multimodal_fusion model (segmentation + learned
object localization head). The model is split inside the backbone:

  Head (sensor side, --front-device):
    1. Captures co-located RGB + radar from a CARLA traffic-light pole.
    2. Builds the 4-channel radar tensor (occupancy, inverse_range,
       radial_velocity, stationary_age) co-registered with the RGB image plane.
    3. Concatenates [RGB, radar] -> 7-channel input.
    4. Runs the fused MobileNetV3 backbone -> dict of intermediate features.
    5. Sends features + per-frame camera-to-world matrix + intrinsics over
       localhost UDP, with zlib feature compression by default.

  Tail (server side, --back-device):
    1. Reconstructs the feature dict.
    2. Runs the LR-ASPP segmentation classifier -> 3-class mask.
    3. Runs the object head -> 11-channel object map.
    4. Decodes object peaks via object_targets.decode_objects, recovering global
       (X, Y) by transforming sensor-relative XYZ through the camera-to-world
       matrix. Projects each predicted 3D OBB to a 2D pixel bbox.
    5. Sends {mask, objects} back over UDP.

  Head again:
    Renders segmentation overlay + 2D bboxes + global-XY/dim/yaw labels on the
    live RGB feed and displays it. This sibling copy also publishes the same
    frame-keyed object results over UDP to
    real_time_spatial_map_server_fusion_object_v1.py so the top-down spatial
    map stays aligned with the rendered camera overlay.

The added --role flag mirrors the OD/segmentation OAI scripts:

  --role loopback
    Run the front and back halves in this one process, preserving the local
    baseline behavior.

  --role front
    Run CARLA sensors and the model front half on the UE/front host. Bind UDP
    sockets to the UE tunnel IP and send features to the remote back half.

  --role back
    Run only the fusion model back half. This role is suitable for the
    oai-perception-rx container because it does not connect to CARLA.

Press q or Esc in the OpenCV view to exit.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import json
import math
import os
import queue
import random
import socket
import subprocess
import sys
import threading
import time
import zlib
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import carla_split_inference_udp_demo as od_demo
import carla_split_inference_udp_data_collect as od_collect
import carla_split_inference_udp_segmentation_demo as seg_demo
import carla_split_inference_udp_segmentation_trained_lraspp_demo as trained_seg_demo
import carla_split_inference_udp_segmentation_trained_lraspp_pole_client as pole_client

# Late imports from the fusion workflow package. These need PYTHONPATH to
# include the workflow root, which the launcher script already arranges.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "pole_lraspp_multimodal_fusion"),
)
from pole_lraspp_multimodal_fusion.model import (  # noqa: E402
    OBJECT_HEAD_CHANNELS,
    build_multitask_fusion_lraspp,
)
from pole_lraspp_multimodal_fusion.object_targets import decode_objects  # noqa: E402
from pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    StationaryTrackAccumulator,
    build_radar_sample,
    radar_raw_to_alt_az_depth_velocity,
)
from pole_lraspp_multimodal_fusion.split_runtime import (  # noqa: E402
    MultimodalLRASPPSplitModel,
)

carla = trained_seg_demo.carla
cv2 = trained_seg_demo.cv2

DEFAULT_WINDOW_NAME = "CARLA Pole Fusion Object Inference"
DEFAULT_TRAFFIC_LIGHT_ID = "14"
DEFAULT_CAMERA_YAW_OFFSET_DEG = 90.0
DEFAULT_CAMERA_PITCH_DEG = -35.0
DEFAULT_SPAWN_RADIUS_METERS = 90.0
DEFAULT_RADAR_CHANNELS = 4
DEFAULT_SPATIAL_MAP_PORT = 39201
SPATIAL_STREAM_SCHEMA = "fusion_object_spatial_map.v1"
DEFAULT_SCENESENSE_RUN_ROOT = Path(__file__).resolve().parent / "metrics_logs" / "scenesense_runs"

VEHICLE_BBOX_COLOR_BGR = (0, 240, 255)

FUSION_METRICS_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "run_id",
    "run_group",
    "stream_id",
    "transport_label",
    "role",
    "frame_id",
    "carla_timestamp",
    "result_received",
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "feature_payload_bytes",
    "feature_payload_bytes_uncompressed",
    "feature_payload_chunks",
    "result_payload_bytes_estimate",
    "result_payload_chunks_estimate",
    "mask_present",
    "segmentation_class_count",
    "object_count",
    "radar_projected_points",
    "spatial_map_enabled",
    "spatial_map_dropped_packets",
    "bind_host",
    "remote_host",
    "camera_source_port",
    "remote_port",
    "remote_source_port",
    "camera_result_port",
    "camera_width",
    "camera_height",
    "model_input_width",
    "model_input_height",
    "quantization_mode",
    "entropy_coder",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run split RGB+radar fusion object localization over a "
            "traffic-light-pole CARLA sensor pair, with intermediate features "
            "transported between model halves over localhost UDP."
        )
    )

    # CARLA connection / world.
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--town",
        default="",
        help=(
            "Deprecated no-op. The client always attaches to the currently "
            "loaded CARLA world and never calls load_world()."
        ),
    )
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for traffic + pedestrians.")

    # Traffic-light pole + sensor mounting (mirrors the segmentation pole client).
    parser.add_argument(
        "--traffic-light-id",
        default=DEFAULT_TRAFFIC_LIGHT_ID,
        help="Traffic-light actor id (or OpenDRIVE id, when CARLA exposes one) to mount near.",
    )
    parser.add_argument(
        "--list-traffic-lights",
        action="store_true",
        help="List available traffic light ids and exit.",
    )
    parser.add_argument(
        "--traffic-light-resolve-retries",
        type=int,
        default=6,
        help=(
            "How many times to retry live CARLA traffic-light actor discovery "
            "before falling back to traffic_lights_data.json."
        ),
    )
    parser.add_argument(
        "--traffic-light-resolve-retry-s",
        type=float,
        default=0.5,
        help="Delay between traffic-light actor discovery retries.",
    )
    parser.add_argument(
        "--camera-location-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="Interpret --camera-x/y/z as pole-local offset or as absolute world location.",
    )
    parser.add_argument("--camera-x", type=float, default=0.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=6.0, help="Pole-relative camera height in meters.")
    parser.add_argument(
        "--camera-yaw",
        type=float,
        default=None,
        help="Absolute camera yaw in degrees. Omit to use traffic-light yaw + --camera-yaw-offset.",
    )
    parser.add_argument(
        "--camera-yaw-offset",
        type=float,
        default=DEFAULT_CAMERA_YAW_OFFSET_DEG,
        help="Yaw offset from the traffic light when --camera-yaw is omitted.",
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=DEFAULT_CAMERA_PITCH_DEG,
        help="Camera pitch in degrees. Negative looks downward.",
    )
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--camera-fov", type=float, default=100.0, help="RGB camera FoV in degrees.")
    parser.add_argument(
        "--camera-resolution",
        choices=["custom", *od_demo.CAMERA_RESOLUTION_PRESETS.keys()],
        default="custom",
    )
    parser.add_argument("--camera-width", type=int, default=854)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0, help="Synchronous sensor tick rate.")
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a camera frame before retrying.",
    )
    parser.add_argument("--camera-warmup-ticks", type=int, default=8)

    # Synchronous mode toggle.
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync-world",
        dest="sync_world",
        action="store_true",
        help="Run CARLA world in synchronous mode while this client is active.",
    )
    sync_group.add_argument(
        "--async-world",
        dest="sync_world",
        action="store_false",
        help="Do not force CARLA synchronous mode.",
    )
    parser.set_defaults(sync_world=True)

    # Radar sensor (defaults match configs/fusion_full_run.yaml).
    parser.add_argument("--radar-range", type=float, default=120.0)
    parser.add_argument("--radar-hfov", type=float, default=100.0, help="Radar horizontal FoV in degrees.")
    parser.add_argument("--radar-vfov", type=float, default=30.0, help="Radar vertical FoV in degrees.")
    parser.add_argument("--radar-points-per-second", type=int, default=5000)
    parser.add_argument(
        "--radar-max-velocity",
        type=float,
        default=20.0,
        help="Max abs velocity used to normalize the radial-velocity raster channel.",
    )
    parser.add_argument(
        "--radar-raster-radius-px",
        type=int,
        default=2,
        help="Disk radius painted at each projected radar point.",
    )
    parser.add_argument(
        "--stationary-velocity-mps",
        type=float,
        default=0.35,
        help="Velocity threshold under which a radar bin is considered stationary.",
    )
    parser.add_argument(
        "--parked-threshold-s",
        type=float,
        default=5.0,
        help="Stationary-age threshold defining a parked label.",
    )
    parser.add_argument("--association-grid-m", type=float, default=1.5)
    parser.add_argument("--max-stale-s", type=float, default=2.0)

    # Background NPCs.
    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=30)
    parser.add_argument("--spawn-radius", type=float, default=DEFAULT_SPAWN_RADIUS_METERS)

    # Fusion checkpoint.
    parser.add_argument(
        "--fusion-checkpoint",
        default="",
        help="Path to a fusion best.pt checkpoint (e.g. .../checkpoints/<trial>/best.pt).",
    )
    parser.add_argument(
        "--fusion-experiment-dir",
        default="",
        help=(
            "Optional pole_lraspp_multimodal_fusion experiment directory. If "
            "--fusion-checkpoint is omitted, manifest.json best_checkpoint is used."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of segmentation classes (3: background/vehicle/person).",
    )
    parser.add_argument(
        "--model-input-width",
        type=int,
        default=0,
        help="Override checkpoint input width. 0 = use checkpoint's stored input_size.",
    )
    parser.add_argument(
        "--model-input-height",
        type=int,
        default=0,
        help="Override checkpoint input height. 0 = use checkpoint's stored input_size.",
    )
    parser.add_argument("--object-hidden-channels", type=int, default=128)

    # Object decode parameters (match configs/fusion_full_run.yaml run-3 values).
    parser.add_argument("--object-score-threshold", type=float, default=0.05)
    parser.add_argument("--object-nms-radius-px", type=int, default=4)
    parser.add_argument("--topk-objects", type=int, default=80)

    # UDP transport.
    parser.add_argument("--camera-source-port", type=int, default=51001)
    parser.add_argument("--remote-port", type=int, default=51002)
    parser.add_argument("--remote-source-port", type=int, default=51003)
    parser.add_argument("--camera-result-port", type=int, default=51004)
    parser.add_argument("--chunk-bytes", type=int, default=60000)
    parser.add_argument("--socket-timeout", type=float, default=2.0)
    parser.add_argument(
        "--quantization-mode",
        choices=od_collect.QUANT_MODE_CHOICES,
        default=od_collect.QUANT_MODE_PER_CHANNEL_UINT8,
    )
    parser.add_argument(
        "--entropy-coder",
        choices=od_collect.ENTROPY_CODER_CHOICES,
        default=od_collect.ENTROPY_CODER_ZLIB,
        help="Entropy coder applied to pickled UDP payloads. Default: zlib.",
    )
    parser.add_argument("--zstd-level", type=int, default=3)

    # Devices + UI.
    parser.add_argument("--front-device", default="auto", help="Head-side device.")
    parser.add_argument("--back-device", default="auto", help="Tail-side device.")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV window.")
    parser.add_argument(
        "--mask-strength",
        type=float,
        default=0.55,
        help="Segmentation mask overlay strength in [0, 1].",
    )
    parser.add_argument(
        "--show-radar-points",
        action="store_true",
        help="Overlay a translucent dot for each projected radar return.",
    )
    parser.add_argument(
        "--max-objects-drawn",
        type=int,
        default=30,
        help="Cap how many object boxes are rendered per frame (sorted by score).",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=0.6,
        help="Seconds to wait for the tail-side result for each frame before skipping it.",
    )
    parser.add_argument(
        "--role",
        choices=("loopback", "front", "back"),
        default="loopback",
        help=(
            "'loopback' runs both halves in one process, 'front' runs CARLA "
            "sensors plus the front half, and 'back' runs only the fusion "
            "model tail."
        ),
    )
    parser.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help=(
            "Local interface address for UDP binds. For OAI: use 10.0.0.2 "
            "on the UE/front host and 0.0.0.0 inside the back-half container."
        ),
    )
    parser.add_argument(
        "--remote-host",
        default=None,
        help=(
            "Peer IP for UDP sends. For OAI: use 192.168.70.140 on the "
            "front host and 10.0.0.2 in the back-half container."
        ),
    )
    parser.add_argument(
        "--back-log-every",
        type=int,
        default=0,
        help=(
            "Back-role debug logging interval in processed frames. 0 disables "
            "packet/progress logs."
        ),
    )

    # Live spatial-map publication. Enabled by default for this sibling copy.
    spatial_stream_group = parser.add_mutually_exclusive_group()
    spatial_stream_group.add_argument(
        "--spatial-map-stream",
        dest="spatial_map_stream",
        action="store_true",
        help="Publish frame-keyed fusion objects to the live spatial-map server.",
    )
    spatial_stream_group.add_argument(
        "--no-spatial-map-stream",
        dest="spatial_map_stream",
        action="store_false",
        help="Disable live spatial-map result publication.",
    )
    parser.set_defaults(spatial_map_stream=True)
    parser.add_argument(
        "--spatial-map-host",
        default="127.0.0.1",
        help="Host running real_time_spatial_map_server_fusion_object_v1.py.",
    )
    parser.add_argument(
        "--spatial-map-port",
        type=int,
        default=DEFAULT_SPATIAL_MAP_PORT,
        help="UDP port used by the live fusion-object spatial-map server.",
    )
    parser.add_argument(
        "--spatial-map-stream-id",
        default="",
        help="Optional unique stream id. Defaults to fusion_tl_<traffic-light-id>.",
    )

    # Experiment logging.
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--enable-run-logging",
        dest="run_logging",
        action="store_true",
        help="Write SceneSense run manifest and per-frame fusion metrics CSV.",
    )
    metrics_group.add_argument(
        "--disable-run-logging",
        dest="run_logging",
        action="store_false",
        help="Disable SceneSense run manifest and per-frame metrics CSV.",
    )
    parser.set_defaults(run_logging=True)
    parser.add_argument(
        "--run-id",
        default=os.environ.get("SCENESENSE_RUN_ID", ""),
        help="Per-process run id. Defaults to SCENESENSE_RUN_ID or a unique timestamp.",
    )
    parser.add_argument(
        "--run-group",
        default=os.environ.get("SCENESENSE_RUN_GROUP", ""),
        help=(
            "Experiment grouping label shared by related streams. Defaults to "
            "SCENESENSE_RUN_GROUP or an automatic coarse timestamp bucket."
        ),
    )
    parser.add_argument(
        "--transport-label",
        default="",
        help="Experiment label such as loopback, single_ue_oai, or multi_ue_oai.",
    )
    parser.add_argument(
        "--metrics-root",
        default=str(DEFAULT_SCENESENSE_RUN_ROOT),
        help="Root directory for SceneSense run folders.",
    )
    parser.add_argument(
        "--metrics-run-dir",
        default=os.environ.get("SCENESENSE_RUN_DIR", ""),
        help="Optional explicit run directory shared by multiple streams.",
    )

    # Run termination.
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 = unlimited.")
    parser.add_argument("--run-duration-s", type=float, default=0.0, help="Stop after this many seconds. 0 = unlimited.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera intrinsics + matrices
# ---------------------------------------------------------------------------


def intrinsics_at(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    return np.array(
        [[f, 0.0, float(width) / 2.0], [0.0, f, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def actor_world_matrix(actor: "carla.Actor") -> np.ndarray:
    return np.array(actor.get_transform().get_matrix(), dtype=np.float64)


def actor_world_inverse_matrix(actor: "carla.Actor") -> np.ndarray:
    return np.array(actor.get_transform().get_inverse_matrix(), dtype=np.float64)


# ---------------------------------------------------------------------------
# Pre-processing the head input (matches FusionPoleMultiTaskDataset)
# ---------------------------------------------------------------------------


def prepare_fusion_input(
    *,
    frame_bgr: np.ndarray,
    radar_tensor_chw: np.ndarray,
    model_size: Tuple[int, int],
    device: torch.device,
    rgb_mean: torch.Tensor,
    rgb_std: torch.Tensor,
) -> torch.Tensor:
    model_w, model_h = int(model_size[0]), int(model_size[1])
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != (model_w, model_h):
        rgb = cv2.resize(rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    rgb_tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    rgb_tensor = (rgb_tensor - rgb_mean) / rgb_std

    radar = radar_tensor_chw
    if radar.shape[1] != model_h or radar.shape[2] != model_w:
        # Each channel resampled with nearest for the binary occupancy channel
        # and bilinear for the continuous channels (range, velocity, age).
        resized = []
        for idx, channel in enumerate(radar):
            interp = cv2.INTER_NEAREST if idx == 0 else cv2.INTER_LINEAR
            resized.append(cv2.resize(channel, (model_w, model_h), interpolation=interp))
        radar = np.stack(resized, axis=0).astype(np.float32)
    radar_tensor = torch.from_numpy(np.ascontiguousarray(radar)).unsqueeze(0).to(device=device, dtype=torch.float32)
    return torch.cat([rgb_tensor, radar_tensor], dim=1)


# ---------------------------------------------------------------------------
# Radar pipeline
# ---------------------------------------------------------------------------


class PoleRadarPipeline:
    """Wraps the CARLA radar sensor + stationary tracker + per-frame raster build.

    Lives on the head side. The tracker state must persist across frames so the
    stationary-age channel grows for parked vehicles, mirroring how the dataset
    was collected during training.
    """

    def __init__(
        self,
        *,
        world: "carla.World",
        transform: "carla.Transform",
        args: argparse.Namespace,
        model_input_size: Tuple[int, int],
    ) -> None:
        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("range", str(float(args.radar_range)))
        bp.set_attribute("horizontal_fov", str(float(args.radar_hfov)))
        bp.set_attribute("vertical_fov", str(float(args.radar_vfov)))
        bp.set_attribute("points_per_second", str(int(args.radar_points_per_second)))
        bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))
        self.sensor: "carla.Actor" = world.spawn_actor(bp, transform)
        self.queue: "queue.Queue[carla.RadarMeasurement]" = queue.Queue(maxsize=2)
        self.sensor.listen(lambda measurement: od_demo.put_latest(self.queue, measurement))

        self.tracker = StationaryTrackAccumulator(
            stationary_velocity_mps=float(args.stationary_velocity_mps),
            parked_threshold_s=float(args.parked_threshold_s),
            association_grid_m=float(args.association_grid_m),
            max_stale_s=float(args.max_stale_s),
        )
        self.model_w, self.model_h = int(model_input_size[0]), int(model_input_size[1])
        self.range_m = float(args.radar_range)
        self.max_abs_velocity = float(args.radar_max_velocity)
        self.parked_threshold_s = float(args.parked_threshold_s)
        self.point_radius_px = int(args.radar_raster_radius_px)

    def get_latest(self, timeout: float) -> Optional["carla.RadarMeasurement"]:
        try:
            return self.queue.get(timeout=float(timeout))
        except queue.Empty:
            return None

    def build_tensor(
        self,
        *,
        measurement: "carla.RadarMeasurement",
        camera_intrinsics: np.ndarray,
        camera_inverse_matrix: np.ndarray,
        frame_time_s: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        detections = radar_raw_to_alt_az_depth_velocity(bytes(measurement.raw_data))
        sensor_matrix = np.array(self.sensor.get_transform().get_matrix(), dtype=np.float64)
        tensor, points, _summary = build_radar_sample(
            detections=detections,
            sensor_matrix=sensor_matrix,
            camera_inverse_matrix=camera_inverse_matrix,
            camera_intrinsics=camera_intrinsics,
            width=self.model_w,
            height=self.model_h,
            frame_time_s=float(frame_time_s),
            tracker=self.tracker,
            max_range_m=self.range_m,
            max_abs_velocity_mps=self.max_abs_velocity,
            parked_threshold_s=self.parked_threshold_s,
            point_radius_px=self.point_radius_px,
        )
        return tensor, points

    def destroy(self) -> None:
        try:
            self.sensor.stop()
        except RuntimeError:
            pass
        try:
            self.sensor.destroy()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Head-side split inference
# ---------------------------------------------------------------------------


class CameraSideFusionInference:
    def __init__(
        self,
        *,
        model: MultimodalLRASPPSplitModel,
        sender: "od_collect.UDPMessageSocket",
        transport: "od_collect.TransportConfig",
        device: torch.device,
        model_input_size: Tuple[int, int],
    ) -> None:
        self.model = model
        self.sender = sender
        self.transport = transport
        self.device = device
        self.model_w, self.model_h = int(model_input_size[0]), int(model_input_size[1])
        self.feature_codecs: Dict[str, object] = OrderedDict()
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # Cache an entropy coder so per-frame compression doesn't re-create it.
        self._probe_coder = transport.make_entropy_coder()

    def process(
        self,
        *,
        frame_id: int,
        frame_bgr: np.ndarray,
        radar_tensor: np.ndarray,
        camera_matrix: np.ndarray,
        camera_intrinsics_input: np.ndarray,
        display_size: Tuple[int, int],
    ) -> Dict[str, object]:
        started = time.perf_counter()
        with torch.inference_mode():
            fused = prepare_fusion_input(
                frame_bgr=frame_bgr,
                radar_tensor_chw=radar_tensor,
                model_size=(self.model_w, self.model_h),
                device=self.device,
                rgb_mean=self.rgb_mean,
                rgb_std=self.rgb_std,
            )
            features = self.model.encode(fused)

        (
            serialized_features,
            payload_bytes_uncompressed,
            _per_level_uncompressed,
            _per_level_compressed,
        ) = od_collect.serialize_feature_maps(
            features,
            self.feature_codecs,
            quantization_mode=self.transport.quantization_mode,
            per_level_compress_probe=False,
            entropy_coder=self._probe_coder,
        )
        payload = {
            "frame_id": int(frame_id),
            "batch_size": int(fused.shape[0]),
            "model_input_size": [int(self.model_w), int(self.model_h)],
            "display_size": [int(display_size[0]), int(display_size[1])],
            "feature_shapes": {
                name: tuple(int(v) for v in tensor.shape) for name, tensor in features.items()
            },
            "features": serialized_features,
            "camera_matrix": camera_matrix.astype(np.float64),
            "camera_intrinsics_input": camera_intrinsics_input.astype(np.float64),
            "camera_sent_perf": time.perf_counter(),
        }
        payload_bytes, payload_chunks = self.sender.send(payload)
        return {
            "front_ms": (time.perf_counter() - started) * 1000.0,
            "payload_bytes": int(payload_bytes),
            "payload_bytes_uncompressed": int(payload_bytes_uncompressed),
            "payload_chunks": int(payload_chunks),
        }


# ---------------------------------------------------------------------------
# Tail-side worker
# ---------------------------------------------------------------------------


class FusionRemoteInferenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        model: MultimodalLRASPPSplitModel,
        receiver: "od_collect.UDPMessageSocket",
        sender: "od_collect.UDPMessageSocket",
        device: torch.device,
        stop_event: threading.Event,
        transport: "od_collect.TransportConfig",
        score_threshold: float,
        nms_radius_px: int,
        topk: int,
        max_objects_drawn: int,
        log_every: int = 0,
        label: str = "fusion-back",
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.receiver = receiver
        self.sender = sender
        self.device = device
        self.stop_event = stop_event
        self.transport = transport
        self.feature_codecs: Dict[str, object] = OrderedDict()
        self.score_threshold = float(score_threshold)
        self.nms_radius_px = int(nms_radius_px)
        self.topk = int(topk)
        self.max_objects_drawn = int(max_objects_drawn)
        self.log_every = max(0, int(log_every))
        self.label = str(label)
        self._processed = 0
        self._last_wait_log = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            payload = self.receiver.receive()
            if payload is None:
                if self.log_every > 0:
                    now = time.time()
                    if now - self._last_wait_log >= 5.0:
                        print(f"[{self.label}] waiting for feature tensors...")
                        self._last_wait_log = now
                continue
            try:
                result = self._run_back_half(payload)
                result_payload_bytes, result_payload_chunks = _estimate_udp_payload(
                    result,
                    chunk_bytes=self.sender.chunk_bytes,
                    transport=self.transport,
                )
                result["result_payload_bytes_estimate"] = int(result_payload_bytes)
                result["result_payload_chunks_estimate"] = int(result_payload_chunks)
                result_bytes, result_chunks = self.sender.send(result)
                self._processed += 1
                if self.log_every > 0 and (
                    self._processed == 1 or self._processed % self.log_every == 0
                ):
                    print(
                        f"[{self.label}] frame={int(payload.get('frame_id', -1))} "
                        f"server_ms={float(result.get('server_ms', 0.0)):.1f} "
                        f"result_bytes={int(result_bytes)} chunks={int(result_chunks)}"
                    )
            except Exception as exc:  # pragma: no cover - runtime path
                print(f"Fusion remote worker error: {exc}", file=sys.stderr)

    def _run_back_half(self, payload: Dict[str, object]) -> Dict[str, object]:
        started = time.perf_counter()
        features = od_collect.deserialize_feature_maps(
            payload["features"],
            self.device,
            batch_size=int(payload.get("batch_size", 1)),
            feature_codecs=self.feature_codecs,
            quantization_mode=self.transport.quantization_mode,
        )
        model_input_size = tuple(int(v) for v in payload["model_input_size"])
        display_w, display_h = (int(v) for v in payload["display_size"])
        camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
        camera_intrinsics_input = np.asarray(payload["camera_intrinsics_input"], dtype=np.float64)
        camera_inverse_matrix = np.linalg.inv(camera_matrix)

        with torch.inference_mode():
            outputs = self.model.decode_outputs(
                features,
                output_size=(int(model_input_size[1]), int(model_input_size[0])),
            )
        seg_logits = outputs["out"]
        mask_input_res = (
            seg_logits.argmax(dim=1).squeeze(0).detach().to("cpu").numpy().astype(np.uint8)
        )
        if mask_input_res.shape != (display_h, display_w):
            mask_display = cv2.resize(
                mask_input_res, (display_w, display_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_display = mask_input_res

        objects: List[Dict[str, float]] = []
        if "object" in outputs:
            raw_predictions = decode_objects(
                outputs["object"],
                camera_matrix=camera_matrix,
                topk=self.topk,
                score_threshold=self.score_threshold,
                nms_radius_px=self.nms_radius_px,
            )
            scale_x = float(display_w) / float(model_input_size[0])
            scale_y = float(display_h) / float(model_input_size[1])
            for prediction in raw_predictions[: self.max_objects_drawn]:
                bbox_xyxy = self._project_obb_to_2d_bbox(
                    prediction=prediction,
                    camera_inverse_matrix=camera_inverse_matrix,
                    intrinsics=camera_intrinsics_input,
                    model_size=model_input_size,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                yaw_deg = math.degrees(
                    math.atan2(float(prediction["yaw_sin"]), float(prediction["yaw_cos"]))
                )
                center_display_x = float(prediction["center_x_px"]) * scale_x
                center_display_y = float(prediction["center_y_px"]) * scale_y
                objects.append(
                    {
                        "score": float(prediction["score"]),
                        "center_x_px": center_display_x,
                        "center_y_px": center_display_y,
                        "world_x": float(prediction["world_x"]),
                        "world_y": float(prediction["world_y"]),
                        "world_z": float(prediction["world_z"]),
                        "size_x": float(prediction["size_x"]),
                        "size_y": float(prediction["size_y"]),
                        "size_z": float(prediction["size_z"]),
                        "yaw_deg": float(yaw_deg),
                        "parked_score": float(prediction["parked_score"]),
                        "radar_support_score": float(prediction["radar_support_score"]),
                        "bbox_xyxy": bbox_xyxy,
                    }
                )

        return {
            "frame_id": int(payload["frame_id"]),
            "camera_sent_perf": float(payload["camera_sent_perf"]),
            "server_ms": (time.perf_counter() - started) * 1000.0,
            "mask": mask_display,
            "objects": objects,
        }

    @staticmethod
    def _project_obb_to_2d_bbox(
        *,
        prediction: Dict[str, float],
        camera_inverse_matrix: np.ndarray,
        intrinsics: np.ndarray,
        model_size: Tuple[int, int],
        scale_x: float,
        scale_y: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        size_x = max(0.05, float(prediction["size_x"]))
        size_y = max(0.05, float(prediction["size_y"]))
        size_z = max(0.05, float(prediction["size_z"]))
        half = np.array(
            [
                [+1, +1, +1],
                [+1, +1, -1],
                [+1, -1, +1],
                [+1, -1, -1],
                [-1, +1, +1],
                [-1, +1, -1],
                [-1, -1, +1],
                [-1, -1, -1],
            ],
            dtype=np.float64,
        ) * np.array([size_x / 2.0, size_y / 2.0, size_z / 2.0], dtype=np.float64)
        yaw_sin = float(prediction["yaw_sin"])
        yaw_cos = float(prediction["yaw_cos"])
        norm = max(1e-6, math.hypot(yaw_sin, yaw_cos))
        yaw_sin /= norm
        yaw_cos /= norm
        rotation = np.array(
            [
                [yaw_cos, -yaw_sin, 0.0],
                [yaw_sin, yaw_cos, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotated = half @ rotation.T
        center_world = np.array(
            [float(prediction["world_x"]), float(prediction["world_y"]), float(prediction["world_z"])],
            dtype=np.float64,
        )
        corners_world = rotated + center_world
        homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
        corners_cam = (camera_inverse_matrix @ homo.T).T[:, :3]

        x = corners_cam[:, 0]
        y = corners_cam[:, 1]
        z = corners_cam[:, 2]
        in_front = x > 0.05
        if not np.any(in_front):
            return None
        x = np.where(in_front, x, np.nan)
        u = intrinsics[0, 2] + (y / x) * intrinsics[0, 0]
        v = intrinsics[1, 2] - (z / x) * intrinsics[1, 1]
        u = u[~np.isnan(u)]
        v = v[~np.isnan(v)]
        if u.size == 0:
            return None
        # Convert from model-input pixel space to display pixel space.
        u *= scale_x
        v *= scale_y
        x0 = float(np.min(u))
        y0 = float(np.min(v))
        x1 = float(np.max(u))
        y1 = float(np.max(v))
        return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Tail-side result store: reuse the segmentation_demo store (frame-keyed dict).
# ---------------------------------------------------------------------------


class CameraResultReceiver(threading.Thread):
    def __init__(
        self,
        *,
        receiver: "od_collect.UDPMessageSocket",
        result_store: seg_demo.SegmentationResultStore,
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
            self.result_store.put(int(payload["frame_id"]), payload)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _resolve_fusion_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.fusion_checkpoint:
        return Path(args.fusion_checkpoint).expanduser().resolve()
    if args.fusion_experiment_dir:
        exp_dir = Path(args.fusion_experiment_dir).expanduser().resolve()
        manifest = exp_dir / "manifest.json"
        if manifest.exists():
            import json

            data = json.loads(manifest.read_text(encoding="utf-8"))
            best = data.get("best_checkpoint")
            if isinstance(best, str) and best:
                return Path(best).expanduser().resolve()
        candidates = sorted(exp_dir.glob("checkpoints/*/best.pt"))
        if not candidates:
            raise FileNotFoundError(f"No best.pt found under {exp_dir}/checkpoints/.")
        return candidates[-1]
    raise ValueError("Provide --fusion-checkpoint PATH or --fusion-experiment-dir PATH.")


def load_fusion_model(
    args: argparse.Namespace, device: torch.device
) -> Tuple[MultimodalLRASPPSplitModel, Tuple[int, int]]:
    checkpoint_path = _resolve_fusion_checkpoint_path(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    radar_channels = int(
        (checkpoint.get("radar_channels") if isinstance(checkpoint, dict) else None)
        or DEFAULT_RADAR_CHANNELS
    )
    object_channels = int(
        (checkpoint.get("object_channels") if isinstance(checkpoint, dict) else None)
        or OBJECT_HEAD_CHANNELS
    )
    fuse_low_into_object_head = bool(
        checkpoint.get("fuse_low_into_object_head") if isinstance(checkpoint, dict) else False
    )
    raw_input_size = (
        checkpoint.get("input_size") if isinstance(checkpoint, dict) else None
    ) or [768, 432]
    if int(args.model_input_width) > 0 and int(args.model_input_height) > 0:
        input_size = (int(args.model_input_width), int(args.model_input_height))
    else:
        input_size = (int(raw_input_size[0]), int(raw_input_size[1]))

    model = build_multitask_fusion_lraspp(
        num_classes=int(args.num_classes),
        radar_channels=radar_channels,
        pretrained=False,
        object_channels=object_channels,
        object_hidden_channels=int(args.object_hidden_channels),
        fuse_low_into_object_head=fuse_low_into_object_head,
        device=device,
    ).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Fusion checkpoint missing {len(missing)} keys (first: {missing[:3]})", file=sys.stderr)
    if unexpected:
        print(f"Fusion checkpoint had {len(unexpected)} unexpected keys (first: {unexpected[:3]})", file=sys.stderr)
    model.eval()
    split_model = MultimodalLRASPPSplitModel(model, device, input_size=input_size)
    print(
        f"Loaded fusion checkpoint {checkpoint_path} "
        f"(radar_channels={radar_channels}, object_channels={object_channels}, "
        f"fuse_low_into_object_head={fuse_low_into_object_head}, "
        f"input_size={input_size[0]}x{input_size[1]})"
    )
    return split_model, input_size


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _draw_overlay_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font_scale: float = 0.52,
    thickness: int = 1,
    fg: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv2.LINE_AA)


def draw_fusion_overlay(
    *,
    frame_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    objects: Sequence[Dict[str, object]],
    radar_points_uv: Optional[np.ndarray],
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    args: argparse.Namespace,
    traffic_light_id: str,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    if mask is not None:
        if mask.shape[:2] != annotated.shape[:2]:
            mask = cv2.resize(
                mask, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        palette = trained_seg_demo.SEGMENTATION_OVERLAY_PALETTE_RGB
        colors_rgb = palette[mask.clip(0, len(palette) - 1)]
        colors_bgr = colors_rgb[:, :, ::-1]
        foreground = mask > 0
        strength = min(1.0, max(0.0, float(args.mask_strength)))
        annotated[foreground] = (
            annotated[foreground].astype(np.float32) * (1.0 - strength)
            + colors_bgr[foreground].astype(np.float32) * strength
        ).astype(np.uint8)

    if bool(args.show_radar_points) and radar_points_uv is not None and radar_points_uv.size:
        h, w = annotated.shape[:2]
        for u, v in radar_points_uv:
            iu, iv = int(round(float(u))), int(round(float(v)))
            if 0 <= iu < w and 0 <= iv < h:
                cv2.circle(annotated, (iu, iv), 3, (0, 255, 200), -1, cv2.LINE_AA)

    h, w = annotated.shape[:2]
    for obj in objects:
        cx = int(np.clip(float(obj["center_x_px"]), 0, w - 1))
        cy = int(np.clip(float(obj["center_y_px"]), 0, h - 1))
        cv2.circle(annotated, (cx, cy), 5, VEHICLE_BBOX_COLOR_BGR, 2, cv2.LINE_AA)
        label_x = min(max(8, cx + 8), w - 1)
        label_y_top = max(18, cy - 30)
        _draw_overlay_text(
            annotated,
            f"score {obj['score']:.2f} yaw {obj['yaw_deg']:+.0f}d "
            f"{('parked' if obj['parked_score'] >= 0.5 else 'moving')}",
            (label_x, label_y_top),
            fg=VEHICLE_BBOX_COLOR_BGR,
        )
        _draw_overlay_text(
            annotated,
            f"world ({obj['world_x']:+.1f}, {obj['world_y']:+.1f}) m",
            (label_x, label_y_top + 16),
            fg=VEHICLE_BBOX_COLOR_BGR,
        )
        _draw_overlay_text(
            annotated,
            f"L {obj['size_x']:.1f}m W {obj['size_y']:.1f}m H {obj['size_z']:.1f}m",
            (label_x, label_y_top + 32),
            fg=VEHICLE_BBOX_COLOR_BGR,
        )

    payload_bytes = max(1, int(front_stats["payload_bytes"]))
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    compression_ratio = payload_bytes_uncompressed / payload_bytes
    lines = [
        f"Pole RGB+Radar fusion | traffic light {traffic_light_id}",
        f"Front half: {float(front_stats['front_ms']):.1f} ms",
        (
            "Feature payload: "
            f"{payload_bytes / 1024.0:.1f} KiB, "
            f"{payload_bytes_uncompressed / 1024.0:.1f} KiB baseline, "
            f"{compression_ratio:.2f}x"
        ),
        f"Detections: {len(objects)}",
    ]
    if remote_stats is not None:
        lines.append(f"Back half: {float(remote_stats['server_ms']):.1f} ms")
        lines.append(f"Round trip: {float(remote_stats['round_trip_ms']):.1f} ms")
    else:
        lines.append("Back half: waiting")
        lines.append("Round trip: waiting")

    y = 28
    for line in lines:
        _draw_overlay_text(annotated, line, (10, y), font_scale=0.56, thickness=2)
        y += 24
    return annotated


# ---------------------------------------------------------------------------
# Live spatial-map publication
# ---------------------------------------------------------------------------


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _carla_transform_payload(transform: "carla.Transform") -> Dict[str, Dict[str, float]]:
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def _segmentation_summary(mask: Optional[np.ndarray]) -> Dict[str, object]:
    if mask is None:
        return {"mask_present": False, "class_counts": {}}

    labels, counts = np.unique(mask.astype(np.int64, copy=False), return_counts=True)
    class_names = {0: "background", 1: "vehicle", 2: "person"}
    return {
        "mask_present": True,
        "class_counts": {
            class_names.get(int(label), f"class_{int(label)}"): int(count)
            for label, count in zip(labels, counts)
        },
    }


def _normalize_spatial_objects(
    objects: Sequence[Dict[str, object]],
    *,
    stream_id: str,
    frame_id: int,
) -> List[Dict[str, object]]:
    normalized = []
    for index, obj in enumerate(objects):
        parked_score = _safe_float(obj.get("parked_score"), 0.0)
        motion_state = "parked" if parked_score >= 0.5 else "moving"
        bbox_xyxy = obj.get("bbox_xyxy")
        if bbox_xyxy is not None:
            try:
                bbox_xyxy = [_safe_float(value) for value in bbox_xyxy]  # type: ignore[assignment]
            except TypeError:
                bbox_xyxy = None

        normalized.append(
            {
                "id": f"{stream_id}:{frame_id}:{index}",
                "type": "Vehicle",
                "motion_state": motion_state,
                "score": _safe_float(obj.get("score"), 0.0),
                "location": {
                    "x": _safe_float(obj.get("world_x"), 0.0),
                    "y": _safe_float(obj.get("world_y"), 0.0),
                    "z": _safe_float(obj.get("world_z"), 0.0),
                },
                "dimensions": {
                    "length": max(0.05, _safe_float(obj.get("size_x"), 0.05)),
                    "width": max(0.05, _safe_float(obj.get("size_y"), 0.05)),
                    "height": max(0.05, _safe_float(obj.get("size_z"), 0.05)),
                },
                "yaw_deg": _safe_float(obj.get("yaw_deg"), 0.0),
                "center_px": {
                    "x": _safe_float(obj.get("center_x_px"), 0.0),
                    "y": _safe_float(obj.get("center_y_px"), 0.0),
                },
                "bbox_xyxy": bbox_xyxy,
                "parked_score": parked_score,
                "radar_support_score": _safe_float(obj.get("radar_support_score"), 0.0),
            }
        )
    return normalized


def _sanitize_path_token(value: object, default: str = "run") -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(value or "").strip()
    ).strip("_")
    return token or default


def _default_transport_label(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "transport_label", "") or "").strip()
    if explicit:
        return explicit
    if args.role == "loopback":
        return "loopback"
    remote = str(getattr(args, "remote_host", "") or "")
    if remote.startswith("192.168.") or remote.startswith("10."):
        return "oai"
    return str(args.role)


def _default_run_group(transport_label: str) -> str:
    now = datetime.now()
    bucket_minute = (now.minute // 10) * 10
    bucket = now.replace(minute=bucket_minute, second=0, microsecond=0)
    return f"{bucket:%Y%m%d_%H%M}_{_sanitize_path_token(transport_label)}"


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _git_status_note() -> str:
    repo_dir = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(repo_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3.0,
            check=False,
        )
    except Exception as exc:
        return f"git_status_unavailable: {exc}"
    if result.returncode != 0:
        return "not_a_git_repository"
    output = result.stdout.strip()
    return output if output else "clean"


def _estimate_udp_payload(
    payload: object,
    *,
    chunk_bytes: int,
    transport: "od_collect.TransportConfig",
) -> Tuple[int, int]:
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = transport.make_entropy_coder().compress(raw)
    max_payload = max(1, int(chunk_bytes) - od_collect.HEADER_STRUCT.size)
    return len(compressed), max(1, math.ceil(len(compressed) / max_payload))


class FusionRunLogger:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        run_id: str,
        run_group: str,
        run_dir: Path,
        stream_id: str,
        transport_label: str,
    ) -> None:
        self.args = args
        self.run_id = run_id
        self.run_group = run_group
        self.run_dir = run_dir
        self.stream_id = stream_id
        self.transport_label = transport_label
        self.stream_token = _sanitize_path_token(stream_id, "stream")
        self.stream_dir = self.run_dir / "streams"
        self.manifest_dir = self.run_dir / "manifests"
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.stream_dir / f"{self.stream_token}_metrics.csv"
        self.manifest_path = self.manifest_dir / f"{self.stream_token}_manifest.json"
        self.config_path = self.manifest_dir / f"{self.stream_token}_resolved_config.json"
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FUSION_METRICS_FIELDS)
        self._writer.writeheader()

    @classmethod
    def from_args(
        cls,
        *,
        args: argparse.Namespace,
        stream_id: str,
        transport_label: str,
    ) -> "FusionRunLogger":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stream_token = _sanitize_path_token(stream_id, "stream")
        run_group = str(args.run_group or "").strip() or _default_run_group(transport_label)
        run_id = str(args.run_id or "").strip() or (
            f"{timestamp}_{_sanitize_path_token(transport_label)}_{stream_token}"
        )
        if str(args.metrics_run_dir or "").strip():
            run_dir = Path(args.metrics_run_dir).expanduser().resolve()
        else:
            run_dir = Path(args.metrics_root).expanduser().resolve() / _sanitize_path_token(run_id)
        return cls(
            args=args,
            run_id=run_id,
            run_group=run_group,
            run_dir=run_dir,
            stream_id=stream_id,
            transport_label=transport_label,
        )

    def write_manifest(
        self,
        *,
        world: "carla.World",
        traffic_light: "carla.Actor",
        model_input_size: Tuple[int, int],
        camera_width: int,
        camera_height: int,
        front_device: torch.device,
        back_device: torch.device,
        checkpoint_path: Path,
    ) -> None:
        try:
            town = world.get_map().name
        except Exception:
            town = ""
        manifest = {
            "schema": "scenesense_fusion_run.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "run_group": self.run_group,
            "stream_id": self.stream_id,
            "transport_label": self.transport_label,
            "script": Path(__file__).name,
            "git_status_note": _git_status_note(),
            "role": str(self.args.role),
            "town": town,
            "checkpoint_path": str(checkpoint_path),
            "front_device": str(front_device),
            "back_device": str(back_device),
            "sensor_placement": "traffic_light_pole",
            "camera": {
                "width": int(camera_width),
                "height": int(camera_height),
                "fov": float(self.args.camera_fov),
                "traffic_light_id": str(self.args.traffic_light_id),
                "traffic_light_actor_id": int(traffic_light.id),
                "x": float(self.args.camera_x),
                "y": float(self.args.camera_y),
                "z": float(self.args.camera_z),
                "pitch": float(self.args.camera_pitch),
                "yaw": None if self.args.camera_yaw is None else float(self.args.camera_yaw),
                "yaw_offset": float(self.args.camera_yaw_offset),
                "roll": float(self.args.camera_roll),
            },
            "model_input_size": [int(model_input_size[0]), int(model_input_size[1])],
            "transport": {
                "bind_host": str(self.args.bind_host),
                "remote_host": str(self.args.remote_host or ""),
                "camera_source_port": int(self.args.camera_source_port),
                "remote_port": int(self.args.remote_port),
                "remote_source_port": int(self.args.remote_source_port),
                "camera_result_port": int(self.args.camera_result_port),
                "quantization_mode": str(self.args.quantization_mode),
                "entropy_coder": str(self.args.entropy_coder),
                "chunk_bytes": int(self.args.chunk_bytes),
            },
            "output_files": {
                "metrics_csv": str(self.csv_path),
                "manifest": str(self.manifest_path),
                "resolved_config": str(self.config_path),
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        resolved_config = _json_safe(vars(self.args))
        if isinstance(resolved_config, dict):
            resolved_config.update(
                {
                    "resolved_run_id": self.run_id,
                    "resolved_run_group": self.run_group,
                    "resolved_run_dir": str(self.run_dir),
                    "resolved_stream_id": self.stream_id,
                    "resolved_transport_label": self.transport_label,
                }
            )
        self.config_path.write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def append(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def build_fusion_metrics_row(
    *,
    args: argparse.Namespace,
    run_logger: FusionRunLogger,
    elapsed_s: float,
    stream_id: str,
    frame_id: int,
    carla_timestamp: float,
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    mask: Optional[np.ndarray],
    objects: Sequence[Dict[str, object]],
    radar_projected_points: int,
    spatial_publisher: Optional["SpatialMapResultPublisher"],
    camera_width: int,
    camera_height: int,
    model_input_size: Tuple[int, int],
) -> Dict[str, object]:
    segmentation = _segmentation_summary(mask)
    remote_host = str(args.remote_host if args.remote_host is not None else args.bind_host)
    return {
        "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_s": float(elapsed_s),
        "run_id": run_logger.run_id,
        "run_group": run_logger.run_group,
        "stream_id": stream_id,
        "transport_label": run_logger.transport_label,
        "role": str(args.role),
        "frame_id": int(frame_id),
        "carla_timestamp": float(carla_timestamp),
        "result_received": remote_stats is not None,
        "front_ms": _safe_float(front_stats.get("front_ms"), 0.0),
        "back_ms": _safe_float((remote_stats or {}).get("server_ms"), float("nan")),
        "round_trip_ms": _safe_float((remote_stats or {}).get("round_trip_ms"), float("nan")),
        "feature_payload_bytes": _safe_int(front_stats.get("payload_bytes"), 0),
        "feature_payload_bytes_uncompressed": _safe_int(
            front_stats.get("payload_bytes_uncompressed"),
            0,
        ),
        "feature_payload_chunks": _safe_int(front_stats.get("payload_chunks"), 0),
        "result_payload_bytes_estimate": _safe_int(
            (remote_stats or {}).get("result_payload_bytes_estimate"),
            0,
        ),
        "result_payload_chunks_estimate": _safe_int(
            (remote_stats or {}).get("result_payload_chunks_estimate"),
            0,
        ),
        "mask_present": bool(segmentation.get("mask_present", False)),
        "segmentation_class_count": len(segmentation.get("class_counts", {})),
        "object_count": len(objects),
        "radar_projected_points": int(radar_projected_points),
        "spatial_map_enabled": spatial_publisher is not None,
        "spatial_map_dropped_packets": (
            int(spatial_publisher.dropped_packets) if spatial_publisher is not None else 0
        ),
        "bind_host": str(args.bind_host),
        "remote_host": remote_host,
        "camera_source_port": int(args.camera_source_port),
        "remote_port": int(args.remote_port),
        "remote_source_port": int(args.remote_source_port),
        "camera_result_port": int(args.camera_result_port),
        "camera_width": int(camera_width),
        "camera_height": int(camera_height),
        "model_input_width": int(model_input_size[0]),
        "model_input_height": int(model_input_size[1]),
        "quantization_mode": str(args.quantization_mode),
        "entropy_coder": str(args.entropy_coder),
    }


class SpatialMapResultPublisher:
    """Background UDP publisher for frame-keyed fusion detections."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        stream_id: str,
        traffic_light_id: str,
        traffic_light_actor_id: int,
        traffic_light_opendrive_id: str,
        camera_width: int,
        camera_height: int,
        camera_fov: float,
    ) -> None:
        self.remote = (str(host), int(port))
        self.stream_id = str(stream_id)
        self.traffic_light_id = str(traffic_light_id)
        self.traffic_light_actor_id = int(traffic_light_actor_id)
        self.traffic_light_opendrive_id = str(traffic_light_opendrive_id or "")
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fov = float(camera_fov)
        self.queue: "queue.Queue[Dict[str, object]]" = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dropped_packets = 0
        self._last_drop_warn = 0.0
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        try:
            self.socket.close()
        except OSError:
            pass

    def publish(
        self,
        *,
        frame_id: int,
        carla_timestamp: float,
        camera_transform: "carla.Transform",
        camera_matrix: np.ndarray,
        objects: Sequence[Dict[str, object]],
        mask: Optional[np.ndarray],
        front_stats: Dict[str, object],
        remote_stats: Optional[Dict[str, object]],
    ) -> None:
        payload = {
            "schema": SPATIAL_STREAM_SCHEMA,
            "source_script": Path(__file__).name,
            "stream_id": self.stream_id,
            "node_id": self.stream_id,
            "traffic_light_id": self.traffic_light_id,
            "traffic_light_actor_id": self.traffic_light_actor_id,
            "traffic_light_opendrive_id": self.traffic_light_opendrive_id,
            "frame_id": int(frame_id),
            "timestamp": time.time(),
            "carla_timestamp": float(carla_timestamp),
            "camera": {
                **_carla_transform_payload(camera_transform),
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": self.camera_fov,
                "matrix": np.asarray(camera_matrix, dtype=np.float64).tolist(),
            },
            "segmentation": _segmentation_summary(mask),
            "objects": _normalize_spatial_objects(
                objects,
                stream_id=self.stream_id,
                frame_id=int(frame_id),
            ),
            "latency": {
                "front_ms": _safe_float(front_stats.get("front_ms"), 0.0),
                "back_ms": _safe_float((remote_stats or {}).get("server_ms"), 0.0),
                "round_trip_ms": _safe_float((remote_stats or {}).get("round_trip_ms"), 0.0),
                "payload_bytes": _safe_int(front_stats.get("payload_bytes"), 0),
                "payload_bytes_uncompressed": _safe_int(
                    front_stats.get("payload_bytes_uncompressed"),
                    0,
                ),
                "payload_chunks": _safe_int(front_stats.get("payload_chunks"), 0),
            },
        }

        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.dropped_packets += 1
            now = time.time()
            if now - self._last_drop_warn >= 1.0:
                print(
                    "[SpatialMap] Publisher queue full; dropping live map "
                    f"packet for frame {frame_id} "
                    f"(total_dropped={self.dropped_packets})."
                )
                self._last_drop_warn = now

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                encoded = json.dumps(
                    payload,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                packet = zlib.compress(encoded, level=1)
                if len(packet) > 65507:
                    print(
                        "[SpatialMap] Dropping oversized live map packet "
                        f"for frame {payload.get('frame_id')}: {len(packet)} bytes."
                    )
                    continue
                self.socket.sendto(packet, self.remote)
            except Exception as exc:  # pragma: no cover - runtime path
                print(f"[SpatialMap] UDP publish failed: {exc}", file=sys.stderr)
            finally:
                self.queue.task_done()


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


def _close_split_runtime(
    *,
    stop_event: threading.Event,
    sockets: Sequence[Optional["od_collect.UDPMessageSocket"]],
    remote_worker: Optional[FusionRemoteInferenceWorker],
    result_receiver: CameraResultReceiver,
) -> None:
    stop_event.set()
    if remote_worker is not None:
        remote_worker.join(timeout=1.0)
    result_receiver.join(timeout=1.0)
    for sock in sockets:
        if sock is None:
            continue
        try:
            sock.close()
        except Exception:
            pass


def _get_preloaded_world(client: "carla.Client", requested_town: object) -> "carla.World":
    town = str(requested_town or "").strip()
    if town:
        print(
            f"[CARLA] Ignoring --town {town!r}; using the already loaded "
            "CARLA world from the running server."
        )
    return client.get_world()


class StaticTrafficLightAnchor:
    """Minimal actor-like pole anchor loaded from traffic_lights_data.json."""

    def __init__(
        self,
        *,
        actor_id: int,
        location: "carla.Location",
        yaw_deg: float,
        opendrive_id: str = "",
    ) -> None:
        self.id = int(actor_id)
        self._opendrive_id = str(opendrive_id or "")
        self._transform = carla.Transform(
            location,
            carla.Rotation(pitch=0.0, yaw=float(yaw_deg), roll=0.0),
        )

    def get_transform(self) -> "carla.Transform":
        return self._transform

    def get_location(self) -> "carla.Location":
        return self._transform.location

    def get_opendrive_id(self) -> str:
        return self._opendrive_id


def _static_anchor_yaw(entry: Dict[str, object]) -> Tuple[float, bool]:
    rotation = entry.get("rotation")
    if isinstance(rotation, dict) and "yaw" in rotation:
        return _safe_float(rotation.get("yaw"), 0.0), True
    if "yaw" in entry:
        return _safe_float(entry.get("yaw"), 0.0), True
    return 0.0, False


def _load_static_traffic_light_anchor(
    requested_id: str,
    args: argparse.Namespace,
) -> Optional[StaticTrafficLightAnchor]:
    path = Path(__file__).resolve().parent / "traffic_lights_data.json"
    if not path.exists():
        return None

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[CARLA] Could not read static traffic-light fallback {path}: {exc}")
        return None

    requested = str(requested_id).strip()
    for entry in rows if isinstance(rows, list) else []:
        if not isinstance(entry, dict):
            continue
        candidates = {str(entry.get("id", "")).strip()}
        opendrive_id = str(entry.get("opendrive_id", "")).strip()
        if opendrive_id:
            candidates.add(opendrive_id)
        if requested not in candidates:
            continue

        location_data = entry.get("location")
        if not isinstance(location_data, dict):
            return None
        yaw_deg, has_yaw = _static_anchor_yaw(entry)
        if not has_yaw and args.camera_yaw is None:
            print(
                "[CARLA] Static traffic-light fallback has no saved yaw. "
                "Using yaw=0 before --camera-yaw-offset; pass --camera-yaw "
                "for exact pointing when using fallback anchors."
            )

        print(
            "[CARLA] Falling back to saved traffic_lights_data.json anchor "
            f"for traffic light {requested!r}. Live CARLA traffic-light actors "
            "were not visible to this client instance."
        )
        return StaticTrafficLightAnchor(
            actor_id=_safe_int(entry.get("id"), 0),
            opendrive_id=opendrive_id,
            yaw_deg=yaw_deg,
            location=carla.Location(
                x=_safe_float(location_data.get("x"), 0.0),
                y=_safe_float(location_data.get("y"), 0.0),
                z=_safe_float(location_data.get("z"), 0.0),
            ),
        )
    return None


def _resolve_traffic_light_with_fallback(
    world: "carla.World",
    args: argparse.Namespace,
) -> "carla.Actor":
    requested_id = str(args.traffic_light_id)
    attempts = max(1, int(args.traffic_light_resolve_retries))
    retry_s = max(0.0, float(args.traffic_light_resolve_retry_s))
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return pole_client.resolve_traffic_light(world, requested_id)
        except ValueError as exc:
            last_error = exc
            try:
                live_count = len(list(world.get_actors().filter("traffic.traffic_light")))
            except Exception:
                live_count = 0
            if live_count > 0:
                raise
            if attempt < attempts:
                print(
                    "[CARLA] No live traffic-light actors visible while resolving "
                    f"{requested_id!r}; retrying {attempt}/{attempts}..."
                )
                time.sleep(retry_s)

    fallback_anchor = _load_static_traffic_light_anchor(requested_id, args)
    if fallback_anchor is not None:
        return fallback_anchor  # type: ignore[return-value]
    if last_error is not None:
        raise last_error
    raise ValueError(f"Traffic light id {requested_id!r} could not be resolved.")


def _transport_config_from_args(args: argparse.Namespace) -> "od_collect.TransportConfig":
    return od_collect.TransportConfig(
        quantization_mode=str(args.quantization_mode),
        entropy_coder_name=str(args.entropy_coder),
        zstd_level=int(args.zstd_level),
        roi_objectness_threshold=0.0,
        bypass_rcnn_transform=False,
    )


def run_back_only(args: argparse.Namespace) -> None:
    """Run only the fusion model back half for the OAI receiver container."""
    back_device = od_demo.resolve_device(args.back_device)
    if back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    back_split_model, _model_input_size = load_fusion_model(args, back_device)
    transport_cfg = _transport_config_from_args(args)
    remote_host = args.remote_host if args.remote_host is not None else args.bind_host

    remote_receiver = od_collect.UDPMessageSocket(
        bind_port=args.remote_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )
    remote_sender = od_collect.UDPMessageSocket(
        bind_port=args.remote_source_port,
        remote_port=args.camera_result_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )

    stop_event = threading.Event()
    remote_worker = FusionRemoteInferenceWorker(
        model=back_split_model,
        receiver=remote_receiver,
        sender=remote_sender,
        device=back_device,
        stop_event=stop_event,
        transport=transport_cfg,
        score_threshold=float(args.object_score_threshold),
        nms_radius_px=int(args.object_nms_radius_px),
        topk=int(args.topk_objects),
        max_objects_drawn=int(args.max_objects_drawn),
        log_every=int(args.back_log_every),
        label=f"fusion-back:{args.remote_port}->{remote_host}:{args.camera_result_port}",
    )
    remote_worker.start()

    print(
        f"[fusion-back] device={back_device} "
        f"recv {args.bind_host}:{args.remote_port}, "
        f"send -> {remote_host}:{args.camera_result_port}"
    )
    print(
        f"[fusion-back] entropy={args.entropy_coder} "
        f"quantization={args.quantization_mode}"
    )
    print("[fusion-back] Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[fusion-back] Interrupted.")
    finally:
        stop_event.set()
        for sock in (remote_receiver, remote_sender):
            try:
                sock.close()
            except OSError:
                pass
        remote_worker.join(timeout=2.0)
        print("[fusion-back] Done.")


def run_client(args: argparse.Namespace) -> None:
    if args.role == "back":
        run_back_only(args)
        return

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    if bool(args.list_traffic_lights):
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = _get_preloaded_world(client, args.town)
        pole_client.list_traffic_lights(world)
        return

    front_device = od_demo.resolve_device(args.front_device)
    back_device = od_demo.resolve_device(args.back_device)
    camera_width, camera_height, camera_resolution_label = od_demo.resolve_camera_dimensions(args)
    gui_enabled = od_demo.has_graphical_display() and not bool(args.headless)

    if front_device.type == "cuda" or back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    front_split_model, model_input_size = load_fusion_model(args, front_device)
    if args.role == "loopback":
        if back_device != front_device:
            back_split_model, _ = load_fusion_model(args, back_device)
        else:
            back_split_model = front_split_model
    else:
        back_split_model = None

    transport_cfg = _transport_config_from_args(args)
    remote_host = args.remote_host if args.remote_host is not None else args.bind_host

    camera_sender = od_collect.UDPMessageSocket(
        bind_port=args.camera_source_port,
        remote_port=args.remote_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )
    remote_receiver = (
        od_collect.UDPMessageSocket(
            bind_port=args.remote_port,
            remote_port=None,
            chunk_bytes=args.chunk_bytes,
            socket_timeout=args.socket_timeout,
            host=args.bind_host,
            entropy_coder=transport_cfg.make_entropy_coder(),
        )
        if args.role == "loopback"
        else None
    )
    remote_sender = (
        od_collect.UDPMessageSocket(
            bind_port=args.remote_source_port,
            remote_port=args.camera_result_port,
            chunk_bytes=args.chunk_bytes,
            socket_timeout=args.socket_timeout,
            host=args.bind_host,
            remote_host=remote_host,
            entropy_coder=transport_cfg.make_entropy_coder(),
        )
        if args.role == "loopback"
        else None
    )
    camera_receiver = od_collect.UDPMessageSocket(
        bind_port=args.camera_result_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )

    stop_event = threading.Event()
    result_store = seg_demo.SegmentationResultStore()
    head_inference = CameraSideFusionInference(
        model=front_split_model,
        sender=camera_sender,
        transport=transport_cfg,
        device=front_device,
        model_input_size=model_input_size,
    )
    remote_worker = (
        FusionRemoteInferenceWorker(
            model=back_split_model,
            receiver=remote_receiver,
            sender=remote_sender,
            device=back_device,
            stop_event=stop_event,
            transport=transport_cfg,
            score_threshold=float(args.object_score_threshold),
            nms_radius_px=int(args.object_nms_radius_px),
            topk=int(args.topk_objects),
            max_objects_drawn=int(args.max_objects_drawn),
            log_every=int(args.back_log_every),
            label=f"fusion-loopback:{args.remote_port}->{remote_host}:{args.camera_result_port}",
        )
        if args.role == "loopback"
        else None
    )
    result_receiver = CameraResultReceiver(
        receiver=camera_receiver,
        result_store=result_store,
        stop_event=stop_event,
    )
    if remote_worker is not None:
        remote_worker.start()
    result_receiver.start()

    split_sockets = (camera_sender, remote_receiver, remote_sender, camera_receiver)
    radar_pipeline: Optional[PoleRadarPipeline] = None
    spatial_publisher: Optional[SpatialMapResultPublisher] = None
    metrics_logger: Optional[FusionRunLogger] = None
    actors: List["carla.Actor"] = []
    checkpoint_path = _resolve_fusion_checkpoint_path(args)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = _get_preloaded_world(client, args.town)
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        try:
            traffic_manager.set_random_device_seed(int(args.seed))
        except RuntimeError:
            pass
        try:
            world.set_pedestrians_seed(int(args.seed))
        except Exception:
            pass

        traffic_light = _resolve_traffic_light_with_fallback(world, args)
        camera_transform = pole_client.build_camera_transform(traffic_light, args)
        anchor_location = traffic_light.get_transform().location
        original_settings = world.get_settings()
    except Exception:
        _close_split_runtime(
            stop_event=stop_event,
            sockets=split_sockets,
            remote_worker=remote_worker,
            result_receiver=result_receiver,
        )
        raise

    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"World: {world.get_map().name}")
    print(f"Traffic light actor id: {traffic_light.id}")
    od_id = pole_client._traffic_light_opendrive_id(traffic_light)
    if od_id:
        print(f"Traffic light OpenDRIVE id: {od_id}")
    print(
        "Pole sensor transform: "
        f"loc=({camera_transform.location.x:.2f}, {camera_transform.location.y:.2f}, "
        f"{camera_transform.location.z:.2f}), "
        f"pitch={camera_transform.rotation.pitch:.1f}, "
        f"yaw={camera_transform.rotation.yaw:.1f}, "
        f"roll={camera_transform.rotation.roll:.1f}"
    )
    print(f"Camera resolution: {camera_width}x{camera_height} ({camera_resolution_label})")
    print(f"Model input: {model_input_size[0]}x{model_input_size[1]}")
    print(f"Front device: {front_device}, back device: {back_device}")
    print(f"Entropy coder: {args.entropy_coder} | Quantization: {args.quantization_mode}")
    print(f"Role: {args.role} | bind-host: {args.bind_host} | remote-host: {remote_host}")
    print(
        "UDP ports: "
        f"camera {args.camera_source_port} -> remote {args.remote_port}, "
        f"remote {args.remote_source_port} -> camera {args.camera_result_port}"
    )

    try:
        if bool(args.sync_world):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / max(0.1, float(args.fps))
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.tick()
        # In --async-world mode this client does NOT touch traffic_manager
        # sync state. The TM is shared across clients via the same --tm-port,
        # so a concurrent --sync-world client would have its TM mode silently
        # flipped to async if we toggled it here.

        background_vehicles = pole_client.spawn_background_vehicles_near(
            client,
            world,
            traffic_manager,
            anchor_location,
            int(args.npc_vehicles),
            float(args.spawn_radius),
        )
        actors.extend(background_vehicles)
        if background_vehicles:
            print(f"Spawned {len(background_vehicles)} background vehicles.")

        pedestrians, pedestrian_controllers = pole_client.spawn_background_pedestrians_near(
            client,
            world,
            anchor_location,
            int(args.npc_pedestrians),
            float(args.spawn_radius),
        )
        actors.extend(pedestrians)
        actors.extend(pedestrian_controllers)
        if pedestrians:
            print(f"Spawned {len(pedestrians)} background pedestrians.")

        camera = world.spawn_actor(
            pole_client._camera_blueprint(world, camera_width, camera_height, args.camera_fov, args.fps),
            camera_transform,
        )
        actors.append(camera)
        camera.listen(lambda image: od_demo.put_latest(image_queue, image))

        radar_pipeline = PoleRadarPipeline(
            world=world,
            transform=camera_transform,
            args=args,
            model_input_size=model_input_size,
        )
        actors.append(radar_pipeline.sensor)

        if bool(args.sync_world):
            first_image = od_demo.warmup_camera_stream(
                world,
                image_queue,
                args.camera_warmup_ticks,
                args.camera_timeout,
            )
        else:
            first_image = image_queue.get(timeout=max(1.0, float(args.camera_timeout)))
        print(f"Pole RGB camera ready on frame {first_image.frame}.")

        intrinsics_input = intrinsics_at(
            int(model_input_size[0]), int(model_input_size[1]), float(args.camera_fov)
        )

        spatial_stream_id = str(args.spatial_map_stream_id).strip() or f"fusion_tl_{traffic_light.id}"
        transport_label = _default_transport_label(args)

        if bool(args.run_logging):
            metrics_logger = FusionRunLogger.from_args(
                args=args,
                stream_id=spatial_stream_id,
                transport_label=transport_label,
            )
            metrics_logger.write_manifest(
                world=world,
                traffic_light=traffic_light,
                model_input_size=model_input_size,
                camera_width=int(camera_width),
                camera_height=int(camera_height),
                front_device=front_device,
                back_device=back_device,
                checkpoint_path=checkpoint_path,
            )
            print(f"[Metrics] Run directory: {metrics_logger.run_dir}")
            print(f"[Metrics] Run group: {metrics_logger.run_group}")
            print(f"[Metrics] Stream CSV: {metrics_logger.csv_path}")

        if bool(args.spatial_map_stream):
            spatial_publisher = SpatialMapResultPublisher(
                host=str(args.spatial_map_host),
                port=int(args.spatial_map_port),
                stream_id=spatial_stream_id,
                traffic_light_id=str(args.traffic_light_id),
                traffic_light_actor_id=int(traffic_light.id),
                traffic_light_opendrive_id=od_id,
                camera_width=int(camera_width),
                camera_height=int(camera_height),
                camera_fov=float(args.camera_fov),
            )
            print(
                "[SpatialMap] Streaming fusion objects to "
                f"{args.spatial_map_host}:{args.spatial_map_port} "
                f"as stream_id={spatial_stream_id}"
            )

        if gui_enabled:
            cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        else:
            print("Headless run active. Press Ctrl+C to stop.")

        start_perf = time.perf_counter()
        processed_frames = 0
        max_measurement_frames = max(0, int(args.max_frames))
        run_duration_s = max(0.0, float(args.run_duration_s))

        while True:
            if bool(args.sync_world):
                world_frame = int(world.tick())
                image = od_demo.wait_for_camera_frame(
                    image_queue,
                    world_frame,
                    float(args.camera_timeout),
                )
            else:
                try:
                    image = image_queue.get(timeout=float(args.camera_timeout))
                except queue.Empty:
                    image = None
            if image is None:
                print(f"Warning: camera frame not received within {args.camera_timeout:.1f}s; retrying.")
                continue

            radar_measurement = radar_pipeline.get_latest(timeout=float(args.camera_timeout))
            if radar_measurement is None:
                print(
                    f"Warning: radar measurement not received within {args.camera_timeout:.1f}s; "
                    "skipping frame."
                )
                continue

            frame_bgr = od_demo.camera_image_to_bgr(image)
            camera_inverse_matrix = actor_world_inverse_matrix(camera)
            radar_tensor, radar_points = radar_pipeline.build_tensor(
                measurement=radar_measurement,
                camera_intrinsics=intrinsics_input,
                camera_inverse_matrix=camera_inverse_matrix,
                frame_time_s=float(image.timestamp),
            )
            camera_matrix = actor_world_matrix(camera)
            front_stats = head_inference.process(
                frame_id=int(image.frame),
                frame_bgr=frame_bgr,
                radar_tensor=radar_tensor,
                camera_matrix=camera_matrix,
                camera_intrinsics_input=intrinsics_input,
                display_size=(int(camera_width), int(camera_height)),
            )

            result = result_store.wait_for(
                int(image.frame),
                float(args.result_timeout),
                tick_callback=None,
                tick_hz=max(0.1, float(args.fps)),
            )
            remote_stats = None
            mask: Optional[np.ndarray] = None
            objects: Sequence[Dict[str, object]] = ()
            if result is not None:
                remote_stats = {
                    "server_ms": float(result["server_ms"]),
                    "round_trip_ms": (time.perf_counter() - float(result["camera_sent_perf"])) * 1000.0,
                    "result_payload_bytes_estimate": int(
                        result.get("result_payload_bytes_estimate", 0)
                    ),
                    "result_payload_chunks_estimate": int(
                        result.get("result_payload_chunks_estimate", 0)
                    ),
                }
                mask = result.get("mask") if isinstance(result.get("mask"), np.ndarray) else None
                if isinstance(result.get("objects"), list):
                    objects = result["objects"]

                if spatial_publisher is not None:
                    spatial_publisher.publish(
                        frame_id=int(image.frame),
                        carla_timestamp=float(image.timestamp),
                        camera_transform=camera.get_transform(),
                        camera_matrix=camera_matrix,
                        objects=objects,
                        mask=mask,
                        front_stats=front_stats,
                        remote_stats=remote_stats,
                    )

            processed_frames += 1
            elapsed_s = time.perf_counter() - start_perf
            if metrics_logger is not None:
                radar_projected_points = 0
                try:
                    radar_projected_points = int(
                        np.count_nonzero(radar_points["valid_projection"].astype(bool))
                    )
                except Exception:
                    radar_projected_points = 0
                metrics_logger.append(
                    build_fusion_metrics_row(
                        args=args,
                        run_logger=metrics_logger,
                        elapsed_s=elapsed_s,
                        stream_id=spatial_stream_id,
                        frame_id=int(image.frame),
                        carla_timestamp=float(image.timestamp),
                        front_stats=front_stats,
                        remote_stats=remote_stats,
                        mask=mask,
                        objects=objects,
                        radar_projected_points=radar_projected_points,
                        spatial_publisher=spatial_publisher,
                        camera_width=int(camera_width),
                        camera_height=int(camera_height),
                        model_input_size=model_input_size,
                    )
                )
            if max_measurement_frames > 0 and processed_frames >= max_measurement_frames:
                print(f"Reached --max-frames={max_measurement_frames}; stopping run.")
                break
            if run_duration_s > 0.0 and elapsed_s >= run_duration_s:
                print(f"Reached --run-duration-s={run_duration_s:.1f}; stopping run.")
                break

            if gui_enabled:
                radar_uv = None
                if bool(args.show_radar_points) and radar_points["valid_projection"].size:
                    valid = radar_points["valid_projection"].astype(bool)
                    if np.any(valid):
                        # Scale projected radar (u,v) from model input grid to display grid.
                        scale_x = float(camera_width) / float(model_input_size[0])
                        scale_y = float(camera_height) / float(model_input_size[1])
                        radar_uv = np.stack(
                            [
                                radar_points["u"][valid] * scale_x,
                                radar_points["v"][valid] * scale_y,
                            ],
                            axis=1,
                        )
                annotated = draw_fusion_overlay(
                    frame_bgr=frame_bgr,
                    mask=mask,
                    objects=objects,
                    radar_points_uv=radar_uv,
                    front_stats=front_stats,
                    remote_stats=remote_stats,
                    args=args,
                    traffic_light_id=str(args.traffic_light_id),
                )
                cv2.imshow(DEFAULT_WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        stop_event.set()
        if metrics_logger is not None:
            metrics_logger.close()
            print(f"[Metrics] Saved stream CSV to {metrics_logger.csv_path}")
        if spatial_publisher is not None:
            spatial_publisher.close()
        if bool(args.sync_world):
            # Only the --sync-world owner restores the shared TM + world sync
            # state; an --async-world client must not toggle TM here either,
            # otherwise it disrupts a concurrent sync-world client mid-run.
            try:
                traffic_manager.set_synchronous_mode(False)
            except (RuntimeError, NameError):
                pass
            try:
                world.apply_settings(original_settings)
            except (RuntimeError, NameError):
                pass

        if radar_pipeline is not None:
            try:
                radar_pipeline.destroy()
            except Exception:
                pass
            actors = [a for a in actors if a is not radar_pipeline.sensor]

        pole_client._destroy_actors(actors)
        _close_split_runtime(
            stop_event=stop_event,
            sockets=split_sockets,
            remote_worker=remote_worker,
            result_receiver=result_receiver,
        )
        if gui_enabled:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_client(args)


if __name__ == "__main__":
    main()

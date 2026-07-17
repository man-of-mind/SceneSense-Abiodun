#!/usr/bin/env python3

"""
Traffic-light-pole / parked-ego RGB+radar fusion client for split-inference
object detection over localhost or the OAI 5G transport path.

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
from collections import OrderedDict, deque
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

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception:  # pragma: no cover - depends on CARLA PythonAPI install layout.
    GlobalRoutePlanner = None

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
DEFAULT_EGO_CAMERA_X = 1.8
DEFAULT_EGO_CAMERA_Y = 0.0
DEFAULT_EGO_CAMERA_Z = 1.55
DEFAULT_EGO_CAMERA_PITCH = -4.0
DEFAULT_EGO_CAMERA_YAW = 0.0
DEFAULT_EGO_CAMERA_ROLL = 0.0
DEFAULT_EGO_RADAR_X = 2.0
DEFAULT_EGO_RADAR_Y = 0.0
DEFAULT_EGO_RADAR_Z = 1.0
DEFAULT_EGO_RADAR_PITCH = 0.0
DEFAULT_EGO_RADAR_YAW = 0.0
DEFAULT_EGO_RADAR_ROLL = 0.0

VEHICLE_BBOX_COLOR_BGR = (0, 240, 255)
PERSON_BBOX_COLOR_BGR = (255, 0, 255)
LEARNED_BBOX_COLOR_BGR = (255, 255, 0)

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
    "transport_round_trip_ms_estimate",
    "total_pipeline_ms_estimate",
    "feature_payload_bytes",
    "feature_payload_bytes_uncompressed",
    "feature_payload_chunks",
    "result_payload_bytes_estimate",
    "result_payload_chunks_estimate",
    "mask_present",
    "segmentation_class_count",
    "gt_camera_available",
    "miou_binary",
    "miou_3class_macro",
    "miou_vehicle_iou",
    "miou_person_iou",
    "gt_vehicle_pixels",
    "gt_person_pixels",
    "object_count",
    "radar_projected_points",
    "ego_speed_mps",
    "tracked_target_actor_id",
    "tracked_target_speed_mps",
    "tracked_gap_m",
    "diagnostic_target_actor_id",
    "diagnostic_target_forward_m",
    "diagnostic_target_lateral_m",
    "diagnostic_target_radar_points",
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

FUSION_OBJECT_PREDICTION_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "run_id",
    "run_group",
    "stream_id",
    "frame_id",
    "object_index",
    "class_name",
    "score",
    "world_x",
    "world_y",
    "world_z",
    "yaw_deg",
    "length_m",
    "width_m",
    "height_m",
    "distance_m",
    "center_x_px",
    "center_y_px",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "parked_score",
    "radar_support_score",
    "radar_support_count",
)

FUSION_OBJECT_GROUND_TRUTH_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "run_id",
    "run_group",
    "stream_id",
    "frame_id",
    "carla_timestamp",
    "actor_id",
    "type_id",
    "role_name",
    "class_name",
    "world_x",
    "world_y",
    "world_z",
    "origin_x",
    "origin_y",
    "origin_z",
    "yaw_deg",
    "length_m",
    "width_m",
    "height_m",
    "distance_m",
    "in_camera_frustum",
    "projected_x",
    "projected_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run split RGB+radar fusion object localization over a "
            "traffic-light-pole or parked ego CARLA sensor pair, with "
            "intermediate features transported between model halves over UDP."
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

    # Sensor platform + mounting.
    parser.add_argument(
        "--sensor-platform",
        choices=("pole", "ego_vehicle"),
        default="pole",
        help=(
            "Sensor host. 'pole' preserves the traffic-light pole baseline; "
            "'ego_vehicle' spawns a parked vehicle and attaches front RGB+radar "
            "sensors to it."
        ),
    )
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

    # Parked ego sensor mounting. These are intentionally separate from the
    # pole camera args so existing pole runbooks keep their defaults.
    parser.add_argument(
        "--ego-vehicle-blueprint",
        default="vehicle.lincoln.mkz",
        help="Vehicle blueprint used when --sensor-platform ego_vehicle.",
    )
    parser.add_argument(
        "--ego-spawn-index",
        type=int,
        default=-1,
        help="Map spawn-point index for the parked ego vehicle. Negative = first available spawn point.",
    )
    parser.add_argument(
        "--ego-role-name",
        default="scenesense_fusion_ego",
        help="CARLA role_name assigned to the parked ego vehicle.",
    )
    parser.add_argument(
        "--ego-freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable parked ego vehicle physics after spawn so it stays fixed.",
    )
    parser.add_argument(
        "--ego-autopilot-speed-difference-pct",
        type=float,
        default=60.0,
        help=(
            "Traffic Manager speed reduction for a moving ego when --no-ego-freeze is used. "
            "Higher values make the ego slower."
        ),
    )
    parser.add_argument(
        "--ego-follow-distance-m",
        type=float,
        default=28.0,
        help="Traffic Manager following distance for a moving ego when --no-ego-freeze is used.",
    )
    parser.add_argument(
        "--ego-ignore-lights-pct",
        type=float,
        default=0.0,
        help="Percent chance the moving ego ignores traffic lights. Keep 0 for realistic visual tests.",
    )
    parser.add_argument(
        "--ego-disable-lane-change",
        action="store_true",
        help="Disable Traffic Manager lane changes for the moving ego.",
    )
    parser.add_argument(
        "--ego-fixed-path-spawn-indices",
        default="",
        help=(
            "Optional comma-separated CARLA spawn indices for a pinned Traffic "
            "Manager route, for example 80,85,91,94,99,110,137,80."
        ),
    )
    parser.add_argument(
        "--ego-fixed-path-progress-csv",
        default="",
        help=(
            "Optional route_progress.csv from a previous good moving run. When set, "
            "the ego reuses those recorded x/y/z points as its pinned path."
        ),
    )
    parser.add_argument("--ego-fixed-path-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ego-fixed-path-min-spacing-m", type=float, default=3.0)
    parser.add_argument(
        "--ego-spawn-forward-offset-m",
        type=float,
        default=0.0,
        help="Move the parked ego spawn forward along its lane heading after selecting the spawn point.",
    )
    parser.add_argument(
        "--ego-spawn-right-offset-m",
        type=float,
        default=0.0,
        help=(
            "Move the parked ego spawn laterally to the right of the lane heading. "
            "Use values around 2.5-3.5 m to push the parked ego toward the curb."
        ),
    )
    parser.add_argument(
        "--ego-spawn-z-offset-m",
        type=float,
        default=0.15,
        help="Vertical lift applied to the parked ego spawn transform to avoid road-mesh collisions.",
    )
    parser.add_argument(
        "--ego-spawn-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Yaw adjustment applied to the parked ego after selecting the spawn point.",
    )
    parser.add_argument("--ego-camera-x", type=float, default=DEFAULT_EGO_CAMERA_X)
    parser.add_argument("--ego-camera-y", type=float, default=DEFAULT_EGO_CAMERA_Y)
    parser.add_argument("--ego-camera-z", type=float, default=DEFAULT_EGO_CAMERA_Z)
    parser.add_argument("--ego-camera-pitch", type=float, default=DEFAULT_EGO_CAMERA_PITCH)
    parser.add_argument("--ego-camera-yaw", type=float, default=DEFAULT_EGO_CAMERA_YAW)
    parser.add_argument("--ego-camera-roll", type=float, default=DEFAULT_EGO_CAMERA_ROLL)
    parser.add_argument("--ego-radar-x", type=float, default=DEFAULT_EGO_RADAR_X)
    parser.add_argument("--ego-radar-y", type=float, default=DEFAULT_EGO_RADAR_Y)
    parser.add_argument("--ego-radar-z", type=float, default=DEFAULT_EGO_RADAR_Z)
    parser.add_argument("--ego-radar-pitch", type=float, default=DEFAULT_EGO_RADAR_PITCH)
    parser.add_argument("--ego-radar-yaw", type=float, default=DEFAULT_EGO_RADAR_YAW)
    parser.add_argument("--ego-radar-roll", type=float, default=DEFAULT_EGO_RADAR_ROLL)

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
        "--radar-temporal-window-frames",
        type=int,
        default=1,
        help=(
            "Max-pool this many recent radar rasters. Default 1 preserves existing live behavior; "
            "the 200k training collection used 2."
        ),
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
    # Control the SPEED DISTRIBUTION of background NPC traffic per run (opportunity-window method): sweep this
    # across runs to populate different target-speed regimes. Negative = faster than the limit (TM convention).
    parser.add_argument("--npc-speed-difference-pct", type=float, default=None,
                        help="TM percentage speed difference applied to every background NPC vehicle "
                             "(negative = faster than limit). None = TM default.")
    parser.add_argument("--npc-ignore-lights-pct", type=float, default=0.0,
                        help="Ignore-traffic-lights percentage for background NPC vehicles, so they sustain "
                             "speed instead of stopping (per-observation speed is still logged and binned).")
    parser.add_argument("--spawn-radius", type=float, default=DEFAULT_SPAWN_RADIUS_METERS)

    # Controlled single target for the staleness / FPS requirement experiment.
    parser.add_argument("--controlled-target", choices=["none", "vehicle", "walker"], default="none",
                        help="Spawn ONE constant-velocity target crossing the FOV at a known speed (skips background NPCs).")
    parser.add_argument("--target-speed-mps", type=float, default=13.4, help="Controlled target constant speed (m/s).")
    parser.add_argument("--target-fwd-dist-m", type=float, default=18.0, help="Distance in front of the camera for the lateral crossing line.")
    parser.add_argument("--target-span-m", type=float, default=36.0, help="Lateral crossing span across the FOV (start = -span/2, moves +span).")
    parser.add_argument("--target-vehicle-filter", default="vehicle.lincoln.mkz", help="Blueprint filter for the controlled vehicle target (CARLA 0.10; falls back to any vehicle).")
    parser.add_argument("--overlay-save-dir", default="", help="If set, periodically save RGB frames here for spawn/tracking sanity-check.")
    parser.add_argument("--overlay-save-every", type=int, default=15, help="Save a sanity frame every N processed frames.")

    # Experiment 3: parked ego with one deterministically placed vehicle. This is
    # deliberately separate from --controlled-target (a free-running crossing
    # used by the older speed/staleness experiments) and --tracked-lead (convoy).
    parser.add_argument(
        "--experiment3-target-profile",
        choices=("none", "centered", "lateral_cycle"),
        default="none",
        help=(
            "Spawn one tagged vehicle relative to a parked ego. centered holds a fixed lateral "
            "offset; lateral_cycle kinematically translates it left-right-left while preserving "
            "forward depth and yaw."
        ),
    )
    parser.add_argument(
        "--experiment3-target-forward-m",
        type=float,
        default=15.0,
        help="Target actor-origin distance ahead of the parked ego origin.",
    )
    parser.add_argument(
        "--experiment3-target-lateral-m",
        type=float,
        default=0.0,
        help="Fixed signed lateral offset for the centered profile (positive = ego-right).",
    )
    parser.add_argument(
        "--experiment3-target-amplitude-m",
        type=float,
        default=8.0,
        help="Half-width of the lateral_cycle path around zero lateral offset.",
    )
    parser.add_argument(
        "--experiment3-target-cycle-frames",
        type=int,
        default=120,
        help="Measured frames in one deterministic left-right-left lateral cycle.",
    )
    parser.add_argument(
        "--experiment3-target-role-name",
        default="scenesense_experiment3_target",
        help="CARLA role_name used to isolate the diagnostic target in GT logs.",
    )
    parser.add_argument(
        "--experiment3-settle-ticks",
        type=int,
        default=30,
        help=(
            "Physics-settling ticks before freezing the Experiment-3 ego and target. "
            "The moving training collector used 30 warm-up ticks before driving."
        ),
    )

    # Convoy scenario: moving ego (use --no-ego-freeze) FOLLOWING one tracked target ahead on its lane, both
    # at the same speed (constant gap -> target stays in view). Staleness = tracked-target speed * Y.
    parser.add_argument("--tracked-lead", choices=["none", "vehicle", "walker"], default="none",
                        help="Spawn ONE tracked target ahead of the moving ego on its lane (convoy).")
    parser.add_argument("--tracked-speed-mps", type=float, default=8.9, help="Tracked target (and matched ego) speed (m/s).")
    parser.add_argument("--tracked-gap-m", type=float, default=15.0, help="Initial gap of the tracked target ahead of the ego (m).")
    parser.add_argument("--tracked-vehicle-filter", default="vehicle.lincoln.mkz", help="Blueprint for the tracked lead vehicle.")
    parser.add_argument(
        "--tracked-motion-control",
        choices=("traffic_manager", "exact"),
        default="traffic_manager",
        help=(
            "Lead/ego motion controller. 'traffic_manager' follows the road; 'exact' gives both "
            "vehicles the same actor-local constant velocity for a short, straight, fixed-gap baseline."
        ),
    )
    parser.add_argument(
        "--tracked-role-name",
        default="scenesense_tracked_lead",
        help="CARLA role_name assigned to the tracked lead so target-only validation can isolate it.",
    )

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
    parser.add_argument(
        "--draw-projected-obb-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the projected-3D-OBB 2D box (from dims/yaw). Deprioritized in the "
        "current model; use --no-draw-projected-obb-box to show only the learned 2D box.",
    )

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
    parser.add_argument(
        "--roi-threshold", type=float, default=0.0,
        help="Front-side ROI drop FRACTION in [0,1): zero the lowest-objectness fraction of "
             "backbone-feature cells (rank-based) before the codec. 0 = off.")
    parser.add_argument(
        "--ae-checkpoint", default="",
        help="Feature-AE checkpoint (ae_bN.pt). If set, the 'high' feature is AE-encoded to the "
             "bottleneck on the front half and AE-decoded on the back half. Composes after ROI drop.")

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
        "--hide-segmentation-mask",
        action="store_true",
        help="Do not render the segmentation mask overlay; keep localization boxes/text visible.",
    )
    parser.add_argument(
        "--show-radar-points",
        action="store_true",
        help="Overlay a translucent dot for each projected radar return.",
    )
    semantic_gt_group = parser.add_mutually_exclusive_group()
    semantic_gt_group.add_argument(
        "--enable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_true",
        help=(
            "Spawn a co-located CARLA semantic-segmentation camera and log "
            "3-class IoU for the returned fusion mask."
        ),
    )
    semantic_gt_group.add_argument(
        "--disable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_false",
        help="Disable semantic-GT camera and segmentation IoU logging.",
    )
    parser.set_defaults(enable_semantic_gt=False)
    parser.add_argument(
        "--max-objects-drawn",
        type=int,
        default=30,
        help="Cap how many object boxes are rendered per frame (sorted by score).",
    )
    parser.add_argument(
        "--hide-object-labels",
        action="store_true",
        help="Do not render per-object score/world/dimension text over localization boxes.",
    )
    parser.add_argument(
        "--object-label-mode",
        choices=("compact", "full", "none"),
        default="compact",
        help=(
            "Object label verbosity. compact shows class/score/distance; full shows "
            "world/dim/yaw details; none hides labels. --hide-object-labels is an alias for none."
        ),
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
        attach_to: Optional["carla.Actor"] = None,
        args: argparse.Namespace,
        model_input_size: Tuple[int, int],
    ) -> None:
        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("range", str(float(args.radar_range)))
        bp.set_attribute("horizontal_fov", str(float(args.radar_hfov)))
        bp.set_attribute("vertical_fov", str(float(args.radar_vfov)))
        bp.set_attribute("points_per_second", str(int(args.radar_points_per_second)))
        bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))
        self.sensor: "carla.Actor" = world.spawn_actor(bp, transform, attach_to=attach_to)
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
        self.tensor_history = deque(
            maxlen=max(1, int(getattr(args, "radar_temporal_window_frames", 1)))
        )

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
        self.tensor_history.append(tensor)
        if self.tensor_history.maxlen > 1 and len(self.tensor_history) > 1:
            tensor = np.maximum.reduce(list(self.tensor_history)).astype(
                np.float32, copy=False
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
            features = _front_compress(self.model, features, tuple(int(v) for v in fused.shape[-2:]))

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
        draw_projected_obb_box: bool = True,
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
        self.draw_projected_obb_box = bool(draw_projected_obb_box)
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
        features = _back_decompress(self.model, features)  # AE-decode bottleneck -> full 'high'
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
                object_class_names=getattr(self.model, "object_class_names", ["vehicle", "person"]),
                predict_bbox2d=bool(getattr(self.model, "object_predict_bbox2d", False)),
            )
            scale_x = float(display_w) / float(model_input_size[0])
            scale_y = float(display_h) / float(model_input_size[1])
            for prediction in raw_predictions[: self.max_objects_drawn]:
                # Projected-3D-OBB box is built from dims/yaw, which are deprioritized
                # in the current model -> unreliable. Off by default; the learned 2D
                # box is the trustworthy one.
                if self.draw_projected_obb_box:
                    bbox_xyxy = self._project_obb_to_2d_bbox(
                        prediction=prediction,
                        camera_inverse_matrix=camera_inverse_matrix,
                        intrinsics=camera_intrinsics_input,
                        model_size=model_input_size,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                else:
                    bbox_xyxy = None
                yaw_deg = math.degrees(
                    math.atan2(float(prediction["yaw_sin"]), float(prediction["yaw_cos"]))
                )
                center_display_x = float(prediction["center_x_px"]) * scale_x
                center_display_y = float(prediction["center_y_px"]) * scale_y
                learned_bbox_xyxy = None
                if all(k in prediction for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")):
                    learned_bbox_xyxy = (
                        float(prediction["bbox_x0"]) * scale_x,
                        float(prediction["bbox_y0"]) * scale_y,
                        float(prediction["bbox_x1"]) * scale_x,
                        float(prediction["bbox_y1"]) * scale_y,
                    )
                distance_m = math.sqrt(
                    float(prediction.get("local_x", 0.0)) ** 2
                    + float(prediction.get("local_y", 0.0)) ** 2
                    + float(prediction.get("local_z", 0.0)) ** 2
                )
                objects.append(
                    {
                        "class_name": str(prediction.get("class_name", "object")),
                        "score": float(prediction["score"]),
                        "center_x_px": center_display_x,
                        "center_y_px": center_display_y,
                        "distance_m": float(distance_m),
                        "local_x": float(prediction.get("local_x", float("nan"))),
                        "local_y": float(prediction.get("local_y", float("nan"))),
                        "local_z": float(prediction.get("local_z", float("nan"))),
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
                        "learned_bbox_xyxy": learned_bbox_xyxy,
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


def _front_compress(split_model, features, out_hw):
    """Front-half split compression, matching training + offline eval:
    (1) rank-based objectness ROI drop of fraction `_roi_threshold`, then (2) AE-encode the 'high'
    feature to the bottleneck. No-op if neither action is configured."""
    import torch.nn.functional as F
    q = float(getattr(split_model, "_roi_threshold", 0.0) or 0.0)
    ae = getattr(split_model, "_ae", None)
    if q <= 0.0 and ae is None:
        return features
    if q > 0.0:
        obj_maps = split_model.decode_object_maps(features, out_hw)
        objness = torch.sigmoid(obj_maps[:, : split_model._n_heat]).amax(dim=1, keepdim=True)
        gated = type(features)()
        for name, feat in features.items():
            pooled = F.adaptive_max_pool2d(objness, feat.shape[-2:]).reshape(-1).float()
            n = pooled.numel()
            k = int(round(q * n))
            keep = torch.ones_like(pooled)
            if k > 0:
                keep[pooled.argsort()[:k]] = 0.0  # drop the k lowest-objectness cells by rank
            gated[name] = feat * keep.reshape(1, 1, feat.shape[-2], feat.shape[-1]).to(feat.dtype)
        features = gated
    if ae is not None:
        features = type(features)((k, (ae.encode(v) if k == "high" else v)) for k, v in features.items())
    return features


def _back_decompress(split_model, features):
    """Back-half: AE-decode the bottleneck 'high' feature to full channels before decode_outputs."""
    ae = getattr(split_model, "_ae", None)
    if ae is None:
        return features
    return type(features)((k, (ae.decode(v) if k == "high" else v)) for k, v in features.items())


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
    object_head_arch = str(
        (checkpoint.get("object_head_arch") if isinstance(checkpoint, dict) else None)
        or "shared"
    )
    object_use_coordconv = bool(
        checkpoint.get("object_use_coordconv") if isinstance(checkpoint, dict) else False
    )
    object_head_depth = int(
        (checkpoint.get("object_head_depth") if isinstance(checkpoint, dict) else None)
        or 2
    )
    object_use_groundplane = bool(
        checkpoint.get("object_use_groundplane_prior") if isinstance(checkpoint, dict) else False
    )
    object_predict_bbox2d = bool(
        checkpoint.get("object_predict_bbox2d") if isinstance(checkpoint, dict) else False
    )
    object_groundplane_params = dict(
        (checkpoint.get("object_groundplane_params") if isinstance(checkpoint, dict) else None)
        or {}
    )
    object_class_names = list(
        (checkpoint.get("object_class_names") if isinstance(checkpoint, dict) else None)
        or ["vehicle", "person"]
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
        head_arch=object_head_arch,
        use_coordconv=object_use_coordconv,
        head_depth=object_head_depth,
        predict_bbox2d=object_predict_bbox2d,
        use_groundplane_prior=object_use_groundplane,
        groundplane_params=object_groundplane_params,
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
    split_model.object_predict_bbox2d = object_predict_bbox2d  # type: ignore[attr-defined]
    split_model.object_class_names = object_class_names  # type: ignore[attr-defined]
    # ROI drop + feature-AE actions (measured on the loopback transport, matching the offline eval).
    split_model._roi_threshold = float(getattr(args, "roi_threshold", 0.0) or 0.0)  # type: ignore[attr-defined]
    split_model._n_heat = int(getattr(model, "heatmap_channels", len(object_class_names)))  # type: ignore[attr-defined]
    split_model._ae = None  # type: ignore[attr-defined]
    ae_ckpt_path = str(getattr(args, "ae_checkpoint", "") or "")
    if ae_ckpt_path:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent / "rl_agent" / "feature_ae"))
        from ae_model import build_ae
        _ck = torch.load(Path(ae_ckpt_path).expanduser(), map_location=device)
        _ae = build_ae(_ck.get("arch", "v1"), int(_ck["in_channels"]), int(_ck["bottleneck"])).to(device)
        _ae.load_state_dict(_ck["ae_state"])
        _ae.eval()
        split_model._ae = _ae  # type: ignore[attr-defined]
        print(f"Loaded feature-AE (bottleneck={_ck['bottleneck']}) for split compression: {ae_ckpt_path}")
    print(
        f"Loaded fusion checkpoint {checkpoint_path} "
        f"(radar_channels={radar_channels}, object_channels={object_channels}, "
        f"fuse_low_into_object_head={fuse_low_into_object_head}, "
        f"object_head_arch={object_head_arch}, "
        f"predict_bbox2d={object_predict_bbox2d}, "
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


def _object_color_bgr(obj: Dict[str, object]) -> Tuple[int, int, int]:
    class_name = str(obj.get("class_name", "")).lower()
    if class_name == "person":
        return PERSON_BBOX_COLOR_BGR
    return VEHICLE_BBOX_COLOR_BGR


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
    if mask is not None and not bool(getattr(args, "hide_segmentation_mask", False)):
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
        color = _object_color_bgr(obj)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _bbox_xyxy_values(obj.get("bbox_xyxy"))
        if all(math.isfinite(v) for v in (bbox_x1, bbox_y1, bbox_x2, bbox_y2)):
            x1 = int(np.clip(round(bbox_x1), 0, w - 1))
            y1 = int(np.clip(round(bbox_y1), 0, h - 1))
            x2 = int(np.clip(round(bbox_x2), 0, w - 1))
            y2 = int(np.clip(round(bbox_y2), 0, h - 1))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        learned_x1, learned_y1, learned_x2, learned_y2 = _bbox_xyxy_values(
            obj.get("learned_bbox_xyxy")
        )
        if all(math.isfinite(v) for v in (learned_x1, learned_y1, learned_x2, learned_y2)):
            x1 = int(np.clip(round(learned_x1), 0, w - 1))
            y1 = int(np.clip(round(learned_y1), 0, h - 1))
            x2 = int(np.clip(round(learned_x2), 0, w - 1))
            y2 = int(np.clip(round(learned_y2), 0, h - 1))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), LEARNED_BBOX_COLOR_BGR, 2, cv2.LINE_AA)
        cx = int(np.clip(float(obj["center_x_px"]), 0, w - 1))
        cy = int(np.clip(float(obj["center_y_px"]), 0, h - 1))
        cv2.circle(annotated, (cx, cy), 5, color, 2, cv2.LINE_AA)
        label_mode = str(getattr(args, "object_label_mode", "compact") or "compact")
        if bool(getattr(args, "hide_object_labels", False)):
            label_mode = "none"
        if label_mode == "none":
            continue
        label_x = min(max(8, cx + 8), w - 1)
        label_y_top = max(18, cy - 30)
        class_name = str(obj.get("class_name", "object"))
        if label_mode == "compact":
            _draw_overlay_text(
                annotated,
                f"{class_name} {float(obj['score']):.2f} | {float(obj.get('distance_m', 0.0)):.1f}m",
                (label_x, label_y_top),
                fg=color,
            )
            continue
        _draw_overlay_text(
            annotated,
            f"{class_name} score {obj['score']:.2f} yaw {obj['yaw_deg']:+.0f}d "
            f"{('parked' if obj['parked_score'] >= 0.5 else 'moving')}",
            (label_x, label_y_top),
            fg=color,
        )
        _draw_overlay_text(
            annotated,
            f"dist {float(obj.get('distance_m', 0.0)):.1f}m | world ({obj['world_x']:+.1f}, {obj['world_y']:+.1f}) m",
            (label_x, label_y_top + 16),
            fg=color,
        )
        _draw_overlay_text(
            annotated,
            f"L {obj['size_x']:.1f}m W {obj['size_y']:.1f}m H {obj['size_z']:.1f}m",
            (label_x, label_y_top + 32),
            fg=color,
        )

    payload_bytes = max(1, int(front_stats["payload_bytes"]))
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    compression_ratio = payload_bytes_uncompressed / payload_bytes
    if str(getattr(args, "sensor_platform", "")) == "ego_vehicle":
        source_kind = (
            "Moving ego RGB+Radar fusion"
            if not bool(getattr(args, "ego_freeze", True))
            else "Parked ego RGB+Radar fusion"
        )
    else:
        source_kind = "Pole RGB+Radar fusion"
    lines = [
        f"{source_kind} | {traffic_light_id}",
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


def _bbox_xyxy_values(value: object) -> Tuple[float, float, float, float]:
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError:
        values = []
    if len(values) != 4:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return tuple(_safe_float(item, float("nan")) for item in values)  # type: ignore[return-value]


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


def _project_world_point_to_image(
    point_world: np.ndarray,
    *,
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[float, float, bool, float]:
    homo = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    point_cam = (camera_inverse_matrix @ homo.T).T[:3]
    depth = float(point_cam[0])
    if depth <= 0.05:
        return (float("nan"), float("nan"), False, depth)
    u = float(intrinsics[0, 2] + (point_cam[1] / depth) * intrinsics[0, 0])
    v = float(intrinsics[1, 2] - (point_cam[2] / depth) * intrinsics[1, 1])
    return (u, v, True, depth)


def _actor_bbox_world_points(actor: "carla.Actor") -> Tuple[np.ndarray, np.ndarray]:
    bbox = actor.bounding_box
    extent = bbox.extent
    center_local = np.array([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64)
    offsets = np.array(
        [
            [+extent.x, +extent.y, +extent.z],
            [+extent.x, +extent.y, -extent.z],
            [+extent.x, -extent.y, +extent.z],
            [+extent.x, -extent.y, -extent.z],
            [-extent.x, +extent.y, +extent.z],
            [-extent.x, +extent.y, -extent.z],
            [-extent.x, -extent.y, +extent.z],
            [-extent.x, -extent.y, -extent.z],
        ],
        dtype=np.float64,
    )
    local_points = center_local[None, :] + offsets
    homo = np.concatenate([local_points, np.ones((local_points.shape[0], 1))], axis=1)
    world_points = (actor_world_matrix(actor) @ homo.T).T[:, :3]
    center_world = (actor_world_matrix(actor) @ np.asarray([*center_local, 1.0], dtype=np.float64).T).T[:3]
    return center_world, world_points


def _project_actor_bbox_to_image(
    actor: "carla.Actor",
    *,
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
    camera_width: int,
    camera_height: int,
) -> Dict[str, object]:
    center_world, corners_world = _actor_bbox_world_points(actor)
    center_u, center_v, center_in_front, _center_depth = _project_world_point_to_image(
        center_world,
        camera_inverse_matrix=camera_inverse_matrix,
        intrinsics=intrinsics,
    )

    homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    corners_cam = (camera_inverse_matrix @ homo.T).T[:, :3]
    depth = corners_cam[:, 0]
    in_front = depth > 0.05
    if not np.any(in_front):
        return {
            "center_world": center_world,
            "projected_x": center_u,
            "projected_y": center_v,
            "in_camera_frustum": False,
            "bbox_xyxy": (float("nan"), float("nan"), float("nan"), float("nan")),
        }

    x = depth[in_front]
    y = corners_cam[in_front, 1]
    z = corners_cam[in_front, 2]
    u = intrinsics[0, 2] + (y / x) * intrinsics[0, 0]
    v = intrinsics[1, 2] - (z / x) * intrinsics[1, 1]
    bbox = (float(np.min(u)), float(np.min(v)), float(np.max(u)), float(np.max(v)))
    intersects = (
        bbox[2] >= 0.0
        and bbox[0] < float(camera_width)
        and bbox[3] >= 0.0
        and bbox[1] < float(camera_height)
    )
    return {
        "center_world": center_world,
        "projected_x": center_u,
        "projected_y": center_v,
        "in_camera_frustum": bool(intersects and (center_in_front or np.any(in_front))),
        "bbox_xyxy": bbox,
    }


def build_vehicle_ground_truth_rows(
    *,
    world: "carla.World",
    frame_id: int,
    elapsed_s: float,
    carla_timestamp: float,
    camera_transform: "carla.Transform",
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
    camera_width: int,
    camera_height: int,
    exclude_actor_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, object]]:
    camera_location = camera_transform.location
    excluded_actor_ids = {int(actor_id) for actor_id in (exclude_actor_ids or ())}
    rows: List[Dict[str, object]] = []
    for actor in world.get_actors().filter("vehicle.*"):
        if int(actor.id) in excluded_actor_ids:
            continue
        try:
            transform = actor.get_transform()
            bbox = actor.bounding_box
            projection = _project_actor_bbox_to_image(
                actor,
                camera_inverse_matrix=camera_inverse_matrix,
                intrinsics=intrinsics,
                camera_width=int(camera_width),
                camera_height=int(camera_height),
            )
        except RuntimeError:
            continue

        center_world = np.asarray(projection["center_world"], dtype=np.float64)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _bbox_xyxy_values(projection["bbox_xyxy"])
        distance_m = math.sqrt(
            (float(center_world[0]) - float(camera_location.x)) ** 2
            + (float(center_world[1]) - float(camera_location.y)) ** 2
            + (float(center_world[2]) - float(camera_location.z)) ** 2
        )
        role_name = ""
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
                "class_name": "vehicle",
                "world_x": float(center_world[0]),
                "world_y": float(center_world[1]),
                "world_z": float(center_world[2]),
                # actor ORIGIN (matches TRAINING GT convention: collect_dataset uses actor.get_location()).
                # world_* above is the bbox CENTER; the model was trained to predict the origin, so validation
                # must compare predictions against origin_* to avoid a spurious ~1 m convention offset.
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


def _segmentation_quality_columns(
    mask: Optional[np.ndarray],
    gt_3class: Optional[np.ndarray],
) -> Dict[str, object]:
    columns: Dict[str, object] = {
        "gt_camera_available": int(gt_3class is not None),
        "miou_binary": float("nan"),
        "miou_3class_macro": float("nan"),
        "miou_vehicle_iou": float("nan"),
        "miou_person_iou": float("nan"),
        "gt_vehicle_pixels": 0,
        "gt_person_pixels": 0,
    }
    if gt_3class is not None:
        columns["gt_vehicle_pixels"] = int(
            np.count_nonzero(gt_3class == trained_seg_demo.CLASS_ID_VEHICLE)
        )
        columns["gt_person_pixels"] = int(
            np.count_nonzero(gt_3class == trained_seg_demo.CLASS_ID_PERSON)
        )
    if mask is None or gt_3class is None:
        return columns
    vehicle_iou, person_iou, macro_iou, binary_iou = trained_seg_demo.compute_3class_iou(
        mask.astype(np.uint8, copy=False),
        gt_3class.astype(np.uint8, copy=False),
    )
    columns["miou_binary"] = float(binary_iou)
    columns["miou_3class_macro"] = float(macro_iou)
    columns["miou_vehicle_iou"] = float(vehicle_iou)
    columns["miou_person_iou"] = float(person_iou)
    return columns


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
                "type": str(obj.get("class_name", "vehicle")).title(),
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
        self.predictions_path = self.stream_dir / f"{self.stream_token}_object_predictions.csv"
        self.ground_truth_path = self.stream_dir / f"{self.stream_token}_object_ground_truth.csv"
        self.manifest_path = self.manifest_dir / f"{self.stream_token}_manifest.json"
        self.config_path = self.manifest_dir / f"{self.stream_token}_resolved_config.json"
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FUSION_METRICS_FIELDS)
        self._writer.writeheader()
        self._prediction_file = self.predictions_path.open("w", newline="", encoding="utf-8")
        self._prediction_writer = csv.DictWriter(
            self._prediction_file,
            fieldnames=FUSION_OBJECT_PREDICTION_FIELDS,
        )
        self._prediction_writer.writeheader()
        self._ground_truth_file = self.ground_truth_path.open("w", newline="", encoding="utf-8")
        self._ground_truth_writer = csv.DictWriter(
            self._ground_truth_file,
            fieldnames=FUSION_OBJECT_GROUND_TRUTH_FIELDS,
        )
        self._ground_truth_writer.writeheader()

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
        anchor_actor: "carla.Actor",
        sensor_placement: str,
        anchor_label: str,
        model_input_size: Tuple[int, int],
        camera_width: int,
        camera_height: int,
        front_device: torch.device,
        back_device: torch.device,
        checkpoint_path: Path,
        tracked_lead_actor: Optional["carla.Actor"] = None,
        experiment3_target_actor: Optional["carla.Actor"] = None,
        camera_actor: Optional["carla.Actor"] = None,
        radar_actor: Optional["carla.Actor"] = None,
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
            "sensor_placement": str(sensor_placement),
            "anchor": {
                "label": str(anchor_label),
                "actor_id": int(anchor_actor.id),
                "type_id": str(getattr(anchor_actor, "type_id", "")),
                "transform": _carla_transform_payload(anchor_actor.get_transform()),
            },
            "tracked_lead": (
                {
                    "actor_id": int(tracked_lead_actor.id),
                    "type_id": str(getattr(tracked_lead_actor, "type_id", "")),
                    "role_name": str(tracked_lead_actor.attributes.get("role_name", "")),
                    "transform": _carla_transform_payload(tracked_lead_actor.get_transform()),
                    "motion_control": str(getattr(self.args, "tracked_motion_control", "")),
                    "commanded_speed_mps": float(getattr(self.args, "tracked_speed_mps", 0.0)),
                    "commanded_gap_m": float(getattr(self.args, "tracked_gap_m", 0.0)),
                }
                if tracked_lead_actor is not None
                else None
            ),
            "experiment3_target": (
                {
                    "actor_id": int(experiment3_target_actor.id),
                    "type_id": str(getattr(experiment3_target_actor, "type_id", "")),
                    "role_name": str(experiment3_target_actor.attributes.get("role_name", "")),
                    "initial_transform": _carla_transform_payload(
                        experiment3_target_actor.get_transform()
                    ),
                    "profile": str(getattr(self.args, "experiment3_target_profile", "none")),
                    "commanded_forward_m": float(
                        getattr(self.args, "experiment3_target_forward_m", 0.0)
                    ),
                    "commanded_lateral_m": float(
                        getattr(self.args, "experiment3_target_lateral_m", 0.0)
                    ),
                    "commanded_amplitude_m": float(
                        getattr(self.args, "experiment3_target_amplitude_m", 0.0)
                    ),
                    "commanded_cycle_frames": int(
                        getattr(self.args, "experiment3_target_cycle_frames", 0)
                    ),
                }
                if experiment3_target_actor is not None
                else None
            ),
            "camera": {
                "width": int(camera_width),
                "height": int(camera_height),
                "fov": float(self.args.camera_fov),
                "traffic_light_id": (
                    str(self.args.traffic_light_id)
                    if str(sensor_placement) == "traffic_light_pole"
                    else ""
                ),
                "traffic_light_actor_id": (
                    int(anchor_actor.id)
                    if str(sensor_placement) == "traffic_light_pole"
                    else None
                ),
                "ego_vehicle_actor_id": (
                    int(anchor_actor.id)
                    if str(sensor_placement) == "ego_vehicle_front"
                    else None
                ),
                "x": float(self.args.camera_x),
                "y": float(self.args.camera_y),
                "z": float(self.args.camera_z),
                "pitch": float(self.args.camera_pitch),
                "yaw": None if self.args.camera_yaw is None else float(self.args.camera_yaw),
                "yaw_offset": float(self.args.camera_yaw_offset),
                "roll": float(self.args.camera_roll),
                "ego_relative_transform": {
                    "x": float(getattr(self.args, "ego_camera_x", 0.0)),
                    "y": float(getattr(self.args, "ego_camera_y", 0.0)),
                    "z": float(getattr(self.args, "ego_camera_z", 0.0)),
                    "pitch": float(getattr(self.args, "ego_camera_pitch", 0.0)),
                    "yaw": float(getattr(self.args, "ego_camera_yaw", 0.0)),
                    "roll": float(getattr(self.args, "ego_camera_roll", 0.0)),
                },
                "actual_world_transform": (
                    _carla_transform_payload(camera_actor.get_transform())
                    if camera_actor is not None
                    else None
                ),
            },
            "radar": {
                "relative_transform": {
                    "x": float(getattr(self.args, "ego_radar_x", 0.0)),
                    "y": float(getattr(self.args, "ego_radar_y", 0.0)),
                    "z": float(getattr(self.args, "ego_radar_z", 0.0)),
                    "pitch": float(getattr(self.args, "ego_radar_pitch", 0.0)),
                    "yaw": float(getattr(self.args, "ego_radar_yaw", 0.0)),
                    "roll": float(getattr(self.args, "ego_radar_roll", 0.0)),
                },
                "range_m": float(self.args.radar_range),
                "hfov": float(self.args.radar_hfov),
                "vfov": float(self.args.radar_vfov),
                "points_per_second": int(self.args.radar_points_per_second),
                "raster_radius_px": int(self.args.radar_raster_radius_px),
                "temporal_window_frames": int(
                    getattr(self.args, "radar_temporal_window_frames", 1)
                ),
                "actual_world_transform": (
                    _carla_transform_payload(radar_actor.get_transform())
                    if radar_actor is not None
                    else None
                ),
            },
            "semantic_gt": {
                "enabled": bool(getattr(self.args, "enable_semantic_gt", False)),
                "metrics": [
                    "miou_binary",
                    "miou_3class_macro",
                    "miou_vehicle_iou",
                    "miou_person_iou",
                ],
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
                "object_predictions_csv": str(self.predictions_path),
                "object_ground_truth_csv": str(self.ground_truth_path),
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

    def append_object_predictions(
        self,
        *,
        elapsed_s: float,
        frame_id: int,
        objects: Sequence[Dict[str, object]],
    ) -> None:
        now = datetime.now().isoformat(timespec="milliseconds")
        for index, obj in enumerate(objects):
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _bbox_xyxy_values(obj.get("bbox_xyxy"))
            learned_bbox_x1, learned_bbox_y1, learned_bbox_x2, learned_bbox_y2 = _bbox_xyxy_values(
                obj.get("learned_bbox_xyxy")
            )
            if all(math.isfinite(v) for v in (learned_bbox_x1, learned_bbox_y1, learned_bbox_x2, learned_bbox_y2)):
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
                    learned_bbox_x1,
                    learned_bbox_y1,
                    learned_bbox_x2,
                    learned_bbox_y2,
                )
            row: Dict[str, object] = {
                "wall_time_iso": now,
                "elapsed_s": float(elapsed_s),
                "run_id": self.run_id,
                "run_group": self.run_group,
                "stream_id": self.stream_id,
                "frame_id": int(frame_id),
                "object_index": int(index),
                "class_name": str(obj.get("class_name", "object")),
                "score": _safe_float(obj.get("score"), float("nan")),
                "world_x": _safe_float(obj.get("world_x"), float("nan")),
                "world_y": _safe_float(obj.get("world_y"), float("nan")),
                "world_z": _safe_float(obj.get("world_z"), float("nan")),
                "yaw_deg": _safe_float(obj.get("yaw_deg"), float("nan")),
                "length_m": _safe_float(obj.get("size_x"), float("nan")),
                "width_m": _safe_float(obj.get("size_y"), float("nan")),
                "height_m": _safe_float(obj.get("size_z"), float("nan")),
                "distance_m": _safe_float(obj.get("distance_m"), float("nan")),
                "center_x_px": _safe_float(obj.get("center_x_px"), float("nan")),
                "center_y_px": _safe_float(obj.get("center_y_px"), float("nan")),
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
                "parked_score": _safe_float(obj.get("parked_score"), float("nan")),
                "radar_support_score": _safe_float(
                    obj.get("radar_support_score"),
                    float("nan"),
                ),
                "radar_support_count": "",
            }
            self._prediction_writer.writerow(
                {field: row.get(field, "") for field in FUSION_OBJECT_PREDICTION_FIELDS}
            )

    def append_object_ground_truth(self, rows: Sequence[Dict[str, object]]) -> None:
        now = datetime.now().isoformat(timespec="milliseconds")
        for row in rows:
            enriched: Dict[str, object] = {
                "wall_time_iso": now,
                "run_id": self.run_id,
                "run_group": self.run_group,
                "stream_id": self.stream_id,
                **row,
            }
            self._ground_truth_writer.writerow(
                {field: enriched.get(field, "") for field in FUSION_OBJECT_GROUND_TRUTH_FIELDS}
            )

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        self._prediction_file.flush()
        self._prediction_file.close()
        self._ground_truth_file.flush()
        self._ground_truth_file.close()


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
        gt_3class: Optional[np.ndarray],
        spatial_publisher: Optional["SpatialMapResultPublisher"],
    camera_width: int,
    camera_height: int,
    model_input_size: Tuple[int, int],
    anchor_actor: Optional["carla.Actor"] = None,
    tracked_lead_actor: Optional["carla.Actor"] = None,
    experiment3_target_actor: Optional["carla.Actor"] = None,
    experiment3_target_radar_points: int = 0,
) -> Dict[str, object]:
    segmentation = _segmentation_summary(mask)
    quality = _segmentation_quality_columns(mask, gt_3class)
    remote_host = str(args.remote_host if args.remote_host is not None else args.bind_host)
    front_ms = _safe_float(front_stats.get("front_ms"), 0.0)
    back_ms = _safe_float((remote_stats or {}).get("server_ms"), float("nan"))
    round_trip_ms = _safe_float((remote_stats or {}).get("round_trip_ms"), float("nan"))
    transport_round_trip_ms = (
        max(0.0, round_trip_ms - back_ms)
        if math.isfinite(round_trip_ms) and math.isfinite(back_ms)
        else float("nan")
    )
    total_pipeline_ms = front_ms + round_trip_ms if math.isfinite(round_trip_ms) else float("nan")
    ego_speed_mps = float("nan")
    tracked_target_speed_mps = float("nan")
    tracked_gap_m = float("nan")
    diagnostic_target_forward_m = float("nan")
    diagnostic_target_lateral_m = float("nan")
    try:
        if anchor_actor is not None and str(args.sensor_platform) == "ego_vehicle":
            velocity = anchor_actor.get_velocity()
            ego_speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if tracked_lead_actor is not None:
            velocity = tracked_lead_actor.get_velocity()
            tracked_target_speed_mps = math.sqrt(
                velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
            )
            if anchor_actor is not None:
                tracked_gap_m = anchor_actor.get_location().distance(
                    tracked_lead_actor.get_location()
                )
        if experiment3_target_actor is not None and anchor_actor is not None:
            diagnostic_target_forward_m, diagnostic_target_lateral_m = (
                _experiment3_target_coordinates(anchor_actor, experiment3_target_actor)
            )
    except RuntimeError:
        pass
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
        "front_ms": front_ms,
        "back_ms": back_ms,
        "round_trip_ms": round_trip_ms,
        "transport_round_trip_ms_estimate": transport_round_trip_ms,
        "total_pipeline_ms_estimate": total_pipeline_ms,
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
        **quality,
        "object_count": len(objects),
        "radar_projected_points": int(radar_projected_points),
        "ego_speed_mps": ego_speed_mps,
        "tracked_target_actor_id": (
            int(tracked_lead_actor.id) if tracked_lead_actor is not None else ""
        ),
        "tracked_target_speed_mps": tracked_target_speed_mps,
        "tracked_gap_m": tracked_gap_m,
        "diagnostic_target_actor_id": (
            int(experiment3_target_actor.id) if experiment3_target_actor is not None else ""
        ),
        "diagnostic_target_forward_m": diagnostic_target_forward_m,
        "diagnostic_target_lateral_m": diagnostic_target_lateral_m,
        "diagnostic_target_radar_points": int(experiment3_target_radar_points),
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


def _relative_transform(
    *,
    x: float,
    y: float,
    z: float,
    pitch: float,
    yaw: float,
    roll: float,
) -> "carla.Transform":
    return carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z)),
        carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=float(roll)),
    )


def _ego_camera_transform(args: argparse.Namespace) -> "carla.Transform":
    return _relative_transform(
        x=float(args.ego_camera_x),
        y=float(args.ego_camera_y),
        z=float(args.ego_camera_z),
        pitch=float(args.ego_camera_pitch),
        yaw=float(args.ego_camera_yaw),
        roll=float(args.ego_camera_roll),
    )


def _ego_radar_transform(args: argparse.Namespace) -> "carla.Transform":
    return _relative_transform(
        x=float(args.ego_radar_x),
        y=float(args.ego_radar_y),
        z=float(args.ego_radar_z),
        pitch=float(args.ego_radar_pitch),
        yaw=float(args.ego_radar_yaw),
        roll=float(args.ego_radar_roll),
    )


def _offset_spawn_transform(
    transform: "carla.Transform",
    *,
    forward_m: float,
    right_m: float,
    z_offset_m: float,
    yaw_offset_deg: float,
) -> "carla.Transform":
    yaw_rad = math.radians(float(transform.rotation.yaw))
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x) + forward_x * float(forward_m) + right_x * float(right_m),
            y=float(transform.location.y) + forward_y * float(forward_m) + right_y * float(right_m),
            z=float(transform.location.z) + float(z_offset_m),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw) + float(yaw_offset_deg),
            roll=float(transform.rotation.roll),
        ),
    )


def _parse_spawn_index_list(text: str) -> List[int]:
    cleaned = str(text or "").replace(";", ",").replace(" ", ",")
    values: List[int] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _copy_location(location: "carla.Location") -> "carla.Location":
    return carla.Location(x=float(location.x), y=float(location.y), z=float(location.z))


def _append_spaced_location(
    route: List["carla.Location"],
    location: "carla.Location",
    min_spacing_m: float,
) -> None:
    candidate = _copy_location(location)
    if not route or route[-1].distance(candidate) >= float(min_spacing_m):
        route.append(candidate)


def _build_fixed_tm_path_from_progress_csv(args: argparse.Namespace) -> List["carla.Location"]:
    progress_csv = str(getattr(args, "ego_fixed_path_progress_csv", "") or "").strip()
    if not progress_csv:
        return []
    path = Path(progress_csv).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Fixed route progress CSV not found: {path}")

    min_spacing_m = max(1.0, float(args.ego_fixed_path_min_spacing_m))
    route: List["carla.Location"] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                location = carla.Location(
                    x=float(row["ego_x"]),
                    y=float(row["ego_y"]),
                    z=float(row.get("ego_z", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            _append_spaced_location(route, location, min_spacing_m)

    if bool(args.ego_fixed_path_loop) and len(route) >= 2 and route[-1].distance(route[0]) > min_spacing_m:
        route.append(_copy_location(route[0]))
    print(
        "Fixed Traffic Manager path: "
        f"source={path}, route_points={len(route)}, min_spacing={min_spacing_m:.1f}m"
    )
    return route


def _build_fixed_tm_path_from_spawn_indices(
    *,
    world: "carla.World",
    args: argparse.Namespace,
) -> List["carla.Location"]:
    indices = _parse_spawn_index_list(str(getattr(args, "ego_fixed_path_spawn_indices", "") or ""))
    if not indices:
        return []

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("Cannot build fixed Traffic Manager path: no map spawn points found.")

    invalid = [idx for idx in indices if idx < 0 or idx >= len(spawn_points)]
    if invalid:
        raise ValueError(
            "Invalid --ego-fixed-path-spawn-indices values "
            f"{invalid}; available index range is 0..{len(spawn_points) - 1}."
        )

    if bool(args.ego_fixed_path_loop) and len(indices) >= 2 and indices[-1] != indices[0]:
        indices = list(indices) + [indices[0]]

    key_points = [spawn_points[idx].location for idx in indices]
    min_spacing_m = max(1.0, float(args.ego_fixed_path_min_spacing_m))
    route: List["carla.Location"] = []
    if len(key_points) < 2:
        return [_copy_location(point) for point in key_points]

    if GlobalRoutePlanner is not None:
        planner = GlobalRoutePlanner(world.get_map(), min_spacing_m)
        for start, end in zip(key_points[:-1], key_points[1:]):
            trace = planner.trace_route(start, end)
            if not trace:
                _append_spaced_location(route, start, min_spacing_m)
                _append_spaced_location(route, end, min_spacing_m)
                continue
            for waypoint, _road_option in trace:
                _append_spaced_location(route, waypoint.transform.location, min_spacing_m)
            _append_spaced_location(route, end, min_spacing_m)
    else:
        for point in key_points:
            _append_spaced_location(route, point, min_spacing_m)

    print(
        "Fixed Traffic Manager path: "
        f"spawn_indices={indices}, route_points={len(route)}, "
        f"planner={'yes' if GlobalRoutePlanner is not None else 'no'}"
    )
    return route


def _build_fixed_tm_path(
    *,
    world: "carla.World",
    args: argparse.Namespace,
) -> List["carla.Location"]:
    path_from_csv = _build_fixed_tm_path_from_progress_csv(args)
    if path_from_csv:
        return path_from_csv
    return _build_fixed_tm_path_from_spawn_indices(world=world, args=args)


def _spawn_parked_ego_vehicle(
    *,
    world: "carla.World",
    args: argparse.Namespace,
) -> "carla.Actor":
    preferred, fell_back = od_demo.resolve_hero_blueprint(world, str(args.ego_vehicle_blueprint))
    if fell_back:
        print(
            f"Requested ego blueprint {args.ego_vehicle_blueprint!r} was not found. "
            f"Falling back to {preferred.id!r}."
        )

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No CARLA spawn points are available for parked ego fusion.")
    if int(args.ego_spawn_index) >= 0:
        index = int(args.ego_spawn_index) % len(spawn_points)
        ordered_spawn_points = [spawn_points[index], *spawn_points[:index], *spawn_points[index + 1 :]]
    else:
        ordered_spawn_points = list(spawn_points)
        random.shuffle(ordered_spawn_points)

    for spawn_point in ordered_spawn_points:
        blueprint = od_demo.get_fresh_vehicle_blueprint(
            world,
            preferred.id,
            str(args.ego_role_name),
        )
        candidate_transform = _offset_spawn_transform(
            spawn_point,
            forward_m=float(args.ego_spawn_forward_offset_m),
            right_m=float(args.ego_spawn_right_offset_m),
            z_offset_m=float(args.ego_spawn_z_offset_m),
            yaw_offset_deg=float(args.ego_spawn_yaw_offset_deg),
        )
        actor = world.try_spawn_actor(blueprint, candidate_transform)
        if actor is None:
            continue
        try:
            actor.set_autopilot(False)
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        except RuntimeError:
            pass
        if bool(args.ego_freeze):
            try:
                if str(getattr(args, "experiment3_target_profile", "none")) != "none":
                    # The moving training collector leaves ego physics enabled for
                    # 30 warm-up ticks. Freezing immediately at the elevated CARLA
                    # spawn transform raises the camera by ~0.7 m at spawn 80 and
                    # breaks visual parity. Let the identical Lincoln settle first.
                    actor.set_simulate_physics(True)
                    for _ in range(max(0, int(args.experiment3_settle_ticks))):
                        world.tick()
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass
        return actor

    raise RuntimeError("Unable to spawn parked ego vehicle at any available spawn point.")


def _experiment3_target_transform(
    *,
    world: "carla.World",
    ego_transform: "carla.Transform",
    forward_m: float,
    lateral_m: float,
) -> "carla.Transform":
    """Return the fixed-pose Experiment-3 target transform.

    Forward/lateral placement is actor-origin to actor-origin in the ego frame.
    X/Y are never road-snapped because doing so would silently change the
    independent variable. The nearest road waypoint is used only for ground Z.
    Target yaw remains equal to ego yaw so lateral position does not change the
    visible vehicle aspect.
    """
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    x = (
        float(ego_transform.location.x)
        + float(forward_m) * float(forward.x)
        + float(lateral_m) * float(right.x)
    )
    y = (
        float(ego_transform.location.y)
        + float(forward_m) * float(forward.y)
        + float(lateral_m) * float(right.y)
    )
    ground_z = float(ego_transform.location.z) - 0.15
    try:
        waypoint = world.get_map().get_waypoint(
            carla.Location(x=x, y=y, z=float(ego_transform.location.z) + 2.0),
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if waypoint is not None:
            ground_z = float(waypoint.transform.location.z)
    except Exception:
        pass
    return carla.Transform(
        carla.Location(x=x, y=y, z=ground_z + 0.30),
        carla.Rotation(
            pitch=float(ego_transform.rotation.pitch),
            yaw=float(ego_transform.rotation.yaw),
            roll=float(ego_transform.rotation.roll),
        ),
    )


def _experiment3_target_coordinates(
    ego_vehicle: "carla.Actor",
    target_actor: "carla.Actor",
) -> Tuple[float, float]:
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    target_location = target_actor.get_location()
    dx = float(target_location.x) - float(ego_location.x)
    dy = float(target_location.y) - float(ego_location.y)
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    return (
        dx * float(forward.x) + dy * float(forward.y),
        dx * float(right.x) + dy * float(right.y),
    )


def _place_experiment3_target(
    *,
    world: "carla.World",
    ego_vehicle: "carla.Actor",
    target_actor: "carla.Actor",
    forward_m: float,
    lateral_m: float,
) -> None:
    target_actor.set_transform(
        _experiment3_target_transform(
            world=world,
            ego_transform=ego_vehicle.get_transform(),
            forward_m=float(forward_m),
            lateral_m=float(lateral_m),
        )
    )


def _spawn_experiment3_target(
    *,
    world: "carla.World",
    ego_vehicle: "carla.Actor",
    vehicle_filter: str,
    role_name: str,
    forward_m: float,
    lateral_m: float,
    settle_ticks: int,
) -> "carla.Actor":
    library = world.get_blueprint_library()
    candidates = list(library.filter(str(vehicle_filter))) or list(library.filter("vehicle.*"))
    if not candidates:
        raise RuntimeError(f"No vehicle blueprint matches {vehicle_filter!r}")
    blueprint = candidates[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    target_transform = _experiment3_target_transform(
        world=world,
        ego_transform=ego_vehicle.get_transform(),
        forward_m=float(forward_m),
        lateral_m=float(lateral_m),
    )
    actor = world.try_spawn_actor(blueprint, target_transform)
    if actor is None:
        raise RuntimeError(
            "Experiment-3 target spawn failed at "
            f"forward={float(forward_m):.2f}m lateral={float(lateral_m):.2f}m"
        )
    try:
        actor.set_autopilot(False)
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        actor.set_simulate_physics(True)
    except RuntimeError:
        pass
    for _ in range(max(0, int(settle_ticks))):
        world.tick()
    # Preserve the physics-settled road height, then restore the exact requested
    # X/Y/yaw before freezing. This matches the training warm-up without allowing
    # the independent forward/lateral variables to drift.
    settled_z = float(actor.get_location().z)
    try:
        actor.set_simulate_physics(False)
        actor.set_transform(
            carla.Transform(
                carla.Location(
                    x=float(target_transform.location.x),
                    y=float(target_transform.location.y),
                    z=settled_z,
                ),
                target_transform.rotation,
            )
        )
    except RuntimeError:
        pass
    world.tick()
    actual_forward, actual_lateral = _experiment3_target_coordinates(ego_vehicle, actor)
    if abs(actual_forward - float(forward_m)) > 0.25 or abs(actual_lateral - float(lateral_m)) > 0.25:
        actor.destroy()
        raise RuntimeError(
            "Experiment-3 target placement verification failed: "
            f"requested=({float(forward_m):.2f},{float(lateral_m):.2f})m "
            f"actual=({actual_forward:.2f},{actual_lateral:.2f})m"
        )
    print(
        "[experiment3-target] "
        f"vehicle {actor.type_id} id={actor.id} role={role_name} "
        f"forward={actual_forward:.3f}m lateral={actual_lateral:.3f}m "
        f"z={actor.get_location().z:.3f}m yaw={actor.get_transform().rotation.yaw:.2f}deg "
        f"physics=settled_then_frozen"
    )
    return actor


def _experiment3_cycle_lateral_offset(args: argparse.Namespace, frame_index: int) -> float:
    """Deterministic left -> right -> left triangular path for measured frames."""
    amplitude = max(0.0, float(args.experiment3_target_amplitude_m))
    total_frames = max(2, int(args.experiment3_target_cycle_frames))
    progress = min(1.0, max(0.0, float(frame_index) / float(total_frames - 1)))
    if progress <= 0.5:
        return -amplitude + 4.0 * amplitude * progress
    return amplitude - 4.0 * amplitude * (progress - 0.5)


def _radar_points_inside_actor_box(
    points_world: np.ndarray,
    actor: Optional["carla.Actor"],
    margin_m: float = 0.35,
) -> int:
    if actor is None or points_world.size == 0:
        return 0
    try:
        inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
        points_h = np.concatenate(
            [np.asarray(points_world, dtype=np.float64), np.ones((points_world.shape[0], 1))],
            axis=1,
        )
        local = (inverse @ points_h.T).T[:, :3]
        bbox = actor.bounding_box
        center = np.asarray([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64)
        extent = np.asarray([bbox.extent.x, bbox.extent.y, bbox.extent.z], dtype=np.float64)
        inside = np.all(np.abs(local - center[None, :]) <= extent[None, :] + float(margin_m), axis=1)
        return int(np.count_nonzero(inside))
    except RuntimeError:
        return 0


def _spawn_controlled_target(*, world, anchor_location, camera_transform, kind, speed_mps,
                             fwd_dist_m, span_m, vehicle_filter):
    """Spawn ONE constant-velocity target crossing the camera FOV laterally at a known speed.
    Vehicle uses enable_constant_velocity (kinematic, exact speed); walker uses WalkerControl.
    Returns (actor, crossing_unit_vector). Placement is kinematic (no road snapping needed) so any
    speed is reachable; z is snapped to nearby road/ground level."""
    import carla, math
    fwd = camera_transform.get_forward_vector()
    right = camera_transform.get_right_vector()
    cam = camera_transform.location
    # start at the FOV center (in front of the camera) so it is guaranteed in-view; then cross laterally.
    start = carla.Location(
        x=cam.x + fwd.x * fwd_dist_m,
        y=cam.y + fwd.y * fwd_dist_m,
        z=cam.z,
    )
    wp = world.get_map().get_waypoint(start, project_to_road=True)
    ground_z = float(wp.transform.location.z) if wp is not None else max(0.0, cam.z - 6.0)
    yaw = math.degrees(math.atan2(right.y, right.x))  # face the +right crossing direction
    bl = world.get_blueprint_library()
    cross = carla.Vector3D(right.x, right.y, 0.0)
    if kind == "vehicle":
        try:
            cbp = list(bl.filter(vehicle_filter)) or list(bl.filter("vehicle.*"))
        except Exception:
            cbp = list(bl.filter("vehicle.*"))
        bp = cbp[0]
        # spawn DIRECTLY at a nearest in-view road point (no teleport -> constant_velocity works reliably)
        cands = []
        for sp in world.get_map().get_spawn_points():
            d = sp.location - cam
            dist = math.hypot(d.x, d.y)
            if 10.0 <= dist <= 32.0 and (d.x * fwd.x + d.y * fwd.y) / max(1e-6, dist) > 0.35:
                cands.append((dist, sp))
        cands.sort(key=lambda t: t[0])
        actor = None
        chosen = None
        for _, sp in cands:
            actor = world.try_spawn_actor(bp, sp)
            if actor is not None:
                chosen = sp
                break
        if actor is None:
            raise RuntimeError(f"no in-view road spawn point ({len(cands)} candidates)")
        world.tick()
        # tangential crossing (perpendicular to the pole->car radial) so it stays ~constant distance / in-view
        d = chosen.location - cam
        dn = math.hypot(d.x, d.y) or 1.0
        tx, ty = -d.y / dn, d.x / dn
        if tx * right.x + ty * right.y < 0.0:
            tx, ty = -tx, -ty
        actor.enable_constant_velocity(carla.Vector3D(tx * speed_mps, ty * speed_mps, 0.0))
        print(f"[controlled-target] vehicle {actor.type_id} id={actor.id} @ {speed_mps:.1f} m/s tangential crossing; "
              f"spawn=({chosen.location.x:.1f},{chosen.location.y:.1f}) dist={cands[0][0]:.0f}m dir=({tx:.2f},{ty:.2f})")
        return actor, (tx, ty)
    else:
        bp = bl.filter("walker.pedestrian.*")[0]
        tf = carla.Transform(carla.Location(start.x, start.y, ground_z + 1.0), carla.Rotation(yaw=yaw))
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            raise RuntimeError("controlled walker spawn failed (adjust --target-fwd-dist-m/--target-span-m)")
        world.tick()
        actor.apply_control(carla.WalkerControl(direction=cross, speed=float(speed_mps)))
        print(f"[controlled-target] walker {actor.type_id} id={actor.id} @ {speed_mps:.1f} m/s, "
              f"start=({start.x:.1f},{start.y:.1f},{ground_z:.1f}), yaw={yaw:.0f}")
        return actor, cross


def _spawn_lead_target(
    *,
    world,
    ego_vehicle,
    traffic_manager,
    tm_port,
    gap_m,
    speed_mps,
    kind,
    vehicle_filter,
    motion_control="traffic_manager",
    role_name="scenesense_tracked_lead",
):
    """Spawn one tagged target ahead of the ego on its lane.

    Traffic-manager mode follows the road. Exact vehicle mode is intentionally for
    short straight baselines: ego and lead receive the same actor-local constant
    velocity so their relative pose does not drift while perception is measured.
    """
    import carla, math
    bl = world.get_blueprint_library()
    ego_wp = world.get_map().get_waypoint(ego_vehicle.get_location())
    ahead = ego_wp.next(max(5.0, float(gap_m)))
    wp = ahead[0] if ahead else ego_wp
    tf = wp.transform
    if kind == "vehicle":
        bp = (list(bl.filter(vehicle_filter)) or list(bl.filter("vehicle.*")))[0]
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", str(role_name))
        exact_motion = str(motion_control) == "exact"
        ego_transform = ego_vehicle.get_transform()
        exact_rotation = ego_transform.rotation
        actor = None
        if exact_motion:
            # Do not use waypoint.next() for the strict baseline: near junctions it
            # can select a branch that is path-distance-correct but not centered on
            # the ego's current heading. Place the lead exactly gap_m along the
            # ego forward vector, then use the road only to recover ground height.
            forward = ego_transform.get_forward_vector()
            desired = carla.Location(
                x=ego_transform.location.x + float(gap_m) * forward.x,
                y=ego_transform.location.y + float(gap_m) * forward.y,
                z=ego_transform.location.z,
            )
            ground_wp = world.get_map().get_waypoint(desired, project_to_road=True)
            loc = carla.Location(
                x=desired.x,
                y=desired.y,
                z=ground_wp.transform.location.z + 0.3,
            )
            actor = world.try_spawn_actor(bp, carla.Transform(loc, exact_rotation))
        else:
            for extra in (0.0, 6.0, 12.0, -4.0):
                nx = wp.next(max(1.0, extra)) if extra > 0 else [wp]
                cand = nx[0] if nx else wp
                loc = carla.Location(
                    cand.transform.location.x,
                    cand.transform.location.y,
                    cand.transform.location.z + 0.3,
                )
                actor = world.try_spawn_actor(bp, carla.Transform(loc, cand.transform.rotation))
                if actor is not None:
                    break
        if actor is None:
            raise RuntimeError("tracked lead vehicle spawn failed (lane occupied)")
        if exact_motion:
            # Freeze immediately so the first initialization tick cannot apply a
            # collision/physics impulse before the paired velocity is installed.
            actor.set_simulate_physics(False)
        world.tick()
        if exact_motion:
            spawn_gap = ego_vehicle.get_location().distance(actor.get_location())
            if abs(spawn_gap - float(gap_m)) > 0.75:
                ego_loc = ego_vehicle.get_location()
                actor_loc = actor.get_location()
                actor.destroy()
                raise RuntimeError(
                    f"exact tracked-lead placement failed: expected {float(gap_m):.2f}m, "
                    f"got {spawn_gap:.2f}m; "
                    f"ego=({ego_loc.x:.2f},{ego_loc.y:.2f},{ego_loc.z:.2f}) "
                    f"requested=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) "
                    f"actor=({actor_loc.x:.2f},{actor_loc.y:.2f},{actor_loc.z:.2f})"
                )
            actor.set_autopilot(False)
            actor.set_simulate_physics(True)
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
            actor.enable_constant_velocity(carla.Vector3D(x=float(speed_mps), y=0.0, z=0.0))
            print(
                f"[tracked-lead] exact vehicle {actor.type_id} id={actor.id} "
                f"~{gap_m:.0f}m ahead @ {speed_mps:.1f} m/s"
            )
        else:
            actor.set_autopilot(True, int(tm_port))
            try:
                traffic_manager.set_desired_speed(actor, float(speed_mps) * 3.6)
            except Exception:
                pass
            try:
                traffic_manager.auto_lane_change(actor, False)
            except Exception:
                pass
            try:
                # Ignore traffic lights so the lead never stops -> convoy stays together (matched-speed ego keeps
                # the gap). Without this the lead halts at lights and the gap blows open (seen previously).
                traffic_manager.ignore_lights_percentage(actor, 100.0)
                traffic_manager.ignore_signs_percentage(actor, 100.0)
            except Exception:
                pass
            print(
                f"[tracked-lead] vehicle {actor.type_id} id={actor.id} "
                f"~{gap_m:.0f}m ahead @ {speed_mps:.1f} m/s (ignore-lights)"
            )
        return actor
    else:
        bp = bl.filter("walker.pedestrian.*")[0]
        loc = carla.Location(tf.location.x, tf.location.y, tf.location.z + 0.8)
        actor = world.try_spawn_actor(bp, carla.Transform(loc, tf.rotation))
        if actor is None:
            raise RuntimeError("tracked lead walker spawn failed")
        world.tick()
        fv = tf.get_forward_vector()
        actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(fv.x, fv.y, 0.0), speed=float(speed_mps)))
        print(f"[tracked-lead] walker id={actor.id} ~{gap_m:.0f}m ahead @ {speed_mps:.1f} m/s")
        return actor


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
        draw_projected_obb_box=bool(args.draw_projected_obb_box),
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
            draw_projected_obb_box=bool(args.draw_projected_obb_box),
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
    sensor_platform = str(args.sensor_platform)
    exact_tracked_convoy = (
        str(getattr(args, "tracked_lead", "none")) != "none"
        and str(getattr(args, "tracked_motion_control", "traffic_manager")) == "exact"
    )
    experiment3_profile = str(getattr(args, "experiment3_target_profile", "none"))
    if exact_tracked_convoy and sensor_platform != "ego_vehicle":
        raise ValueError("--tracked-motion-control exact requires --sensor-platform ego_vehicle")
    if exact_tracked_convoy and bool(args.ego_freeze):
        raise ValueError("--tracked-motion-control exact requires --no-ego-freeze")
    if exact_tracked_convoy and str(args.tracked_lead) != "vehicle":
        raise ValueError("--tracked-motion-control exact currently supports --tracked-lead vehicle only")
    if experiment3_profile != "none" and sensor_platform != "ego_vehicle":
        raise ValueError("--experiment3-target-profile requires --sensor-platform ego_vehicle")
    if experiment3_profile != "none" and not bool(args.ego_freeze):
        raise ValueError("--experiment3-target-profile requires a parked ego (keep --ego-freeze)")
    if experiment3_profile != "none" and (
        str(getattr(args, "controlled_target", "none")) != "none"
        or str(getattr(args, "tracked_lead", "none")) != "none"
    ):
        raise ValueError(
            "--experiment3-target-profile is mutually exclusive with --controlled-target/--tracked-lead"
        )
    traffic_light: Optional["carla.Actor"] = None
    ego_vehicle: Optional["carla.Actor"] = None
    tracked_lead_actor: Optional["carla.Actor"] = None
    experiment3_target_actor: Optional["carla.Actor"] = None
    anchor_actor: Optional["carla.Actor"] = None
    anchor_location: Optional["carla.Location"] = None
    camera_attach_to: Optional["carla.Actor"] = None
    radar_attach_to: Optional["carla.Actor"] = None
    od_id = ""

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

        if sensor_platform == "pole":
            traffic_light = _resolve_traffic_light_with_fallback(world, args)
            camera_transform = pole_client.build_camera_transform(traffic_light, args)
            radar_transform = camera_transform
            anchor_actor = traffic_light
            anchor_location = traffic_light.get_transform().location
            od_id = pole_client._traffic_light_opendrive_id(traffic_light)
        else:
            camera_transform = _ego_camera_transform(args)
            radar_transform = _ego_radar_transform(args)
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
    gt_queue: Optional["queue.Queue[carla.Image]"] = None
    gt_camera: Optional["carla.Actor"] = None
    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"World: {world.get_map().name}")
    print(f"Sensor platform: {sensor_platform}")
    if sensor_platform == "pole" and traffic_light is not None:
        print(f"Traffic light actor id: {traffic_light.id}")
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
    else:
        print(
            "Parked ego relative camera transform: "
            f"loc=({camera_transform.location.x:.2f}, {camera_transform.location.y:.2f}, "
            f"{camera_transform.location.z:.2f}), "
            f"pitch={camera_transform.rotation.pitch:.1f}, "
            f"yaw={camera_transform.rotation.yaw:.1f}, "
            f"roll={camera_transform.rotation.roll:.1f}"
        )
        print(
            "Parked ego relative radar transform: "
            f"loc=({radar_transform.location.x:.2f}, {radar_transform.location.y:.2f}, "
            f"{radar_transform.location.z:.2f}), "
            f"pitch={radar_transform.rotation.pitch:.1f}, "
            f"yaw={radar_transform.rotation.yaw:.1f}, "
            f"roll={radar_transform.rotation.roll:.1f}"
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

        if sensor_platform == "ego_vehicle":
            ego_vehicle = _spawn_parked_ego_vehicle(world=world, args=args)
            actors.append(ego_vehicle)
            # CARLA may report an actor transform at (0,0,0) until the first
            # world tick after spawn. Initialize it before reading anchor
            # position or placing a tracked lead relative to the ego.
            world.tick()
            anchor_actor = ego_vehicle
            anchor_location = ego_vehicle.get_location()
            camera_attach_to = ego_vehicle
            radar_attach_to = ego_vehicle
            if not bool(args.ego_freeze) and not exact_tracked_convoy:
                try:
                    try:
                        traffic_manager.set_global_distance_to_leading_vehicle(
                            max(2.0, float(args.ego_follow_distance_m))
                        )
                    except Exception:
                        pass
                    traffic_manager.ignore_lights_percentage(
                        ego_vehicle,
                        max(0.0, min(100.0, float(args.ego_ignore_lights_pct))),
                    )
                    traffic_manager.vehicle_percentage_speed_difference(
                        ego_vehicle,
                        float(args.ego_autopilot_speed_difference_pct),
                    )
                    try:
                        traffic_manager.distance_to_leading_vehicle(
                            ego_vehicle,
                            max(2.0, float(args.ego_follow_distance_m)),
                        )
                    except Exception:
                        pass
                    if bool(args.ego_disable_lane_change):
                        traffic_manager.auto_lane_change(ego_vehicle, False)
                    ego_vehicle.set_autopilot(True, int(args.tm_port))
                    fixed_path = _build_fixed_tm_path(world=world, args=args)
                    if fixed_path:
                        traffic_manager.set_path(ego_vehicle, list(fixed_path))
                    print(
                        "Moving ego autopilot enabled: "
                        f"speed_diff={float(args.ego_autopilot_speed_difference_pct):.1f}%, "
                        f"follow={float(args.ego_follow_distance_m):.1f}m, "
                        f"ignore_lights={float(args.ego_ignore_lights_pct):.1f}%, "
                        f"lane_change={not bool(args.ego_disable_lane_change)}, "
                        f"fixed_path_points={len(fixed_path)}"
                    )
                except RuntimeError as exc:
                    print(f"WARNING: Could not enable ego autopilot: {exc}", file=sys.stderr)
            elif exact_tracked_convoy:
                print(
                    "Moving ego exact-velocity mode selected; Traffic Manager is disabled "
                    "for the ego/lead pair."
                )
            print(
                ("Moving ego vehicle: " if not bool(args.ego_freeze) else "Parked ego vehicle: ")
                + (
                f"id={ego_vehicle.id}, type={ego_vehicle.type_id}, "
                f"spawn_index={int(args.ego_spawn_index)}, "
                f"freeze={bool(args.ego_freeze)}"
                )
            )
            # tracked target ahead (works for BOTH parked and moving ego); match ego speed only if moving
            if str(getattr(args, "tracked_lead", "none")) != "none":
                if not bool(args.ego_freeze):
                    try:
                        traffic_manager.set_desired_speed(ego_vehicle, float(args.tracked_speed_mps) * 3.6)
                    except Exception:
                        pass
                tracked_lead_actor = _spawn_lead_target(
                    world=world, ego_vehicle=ego_vehicle, traffic_manager=traffic_manager,
                    tm_port=int(args.tm_port), gap_m=float(args.tracked_gap_m),
                    speed_mps=float(args.tracked_speed_mps), kind=str(args.tracked_lead),
                    vehicle_filter=str(args.tracked_vehicle_filter),
                    motion_control=str(args.tracked_motion_control),
                    role_name=str(args.tracked_role_name))
                actors.append(tracked_lead_actor)
                if exact_tracked_convoy:
                    ego_vehicle.set_autopilot(False)
                    ego_vehicle.set_simulate_physics(True)
                    ego_vehicle.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
                    )
                    ego_vehicle.enable_constant_velocity(
                        carla.Vector3D(x=float(args.tracked_speed_mps), y=0.0, z=0.0)
                    )
                    world.tick()
                    gap_now = ego_vehicle.get_location().distance(tracked_lead_actor.get_location())
                    if abs(gap_now - float(args.tracked_gap_m)) > 0.75:
                        raise RuntimeError(
                            "exact convoy preflight failed after velocity enable: "
                            f"expected {float(args.tracked_gap_m):.2f}m, got {gap_now:.2f}m"
                        )
                    print(
                        f"[tracked-convoy] exact ego/lead velocity={float(args.tracked_speed_mps):.2f}m/s "
                        f"initial_gap={gap_now:.2f}m"
                    )

            if experiment3_profile != "none":
                initial_lateral_m = (
                    _experiment3_cycle_lateral_offset(args, 0)
                    if experiment3_profile == "lateral_cycle"
                    else float(args.experiment3_target_lateral_m)
                )
                experiment3_target_actor = _spawn_experiment3_target(
                    world=world,
                    ego_vehicle=ego_vehicle,
                    vehicle_filter=str(args.target_vehicle_filter),
                    role_name=str(args.experiment3_target_role_name),
                    forward_m=float(args.experiment3_target_forward_m),
                    lateral_m=float(initial_lateral_m),
                    settle_ticks=int(args.experiment3_settle_ticks),
                )
                actors.append(experiment3_target_actor)

        if anchor_actor is None or anchor_location is None:
            raise RuntimeError("Sensor anchor was not initialized.")

        controlled_target_actor = None
        _skip_background = (
            str(getattr(args, "controlled_target", "none")) != "none"
            or str(getattr(args, "tracked_lead", "none")) != "none"
            or experiment3_profile != "none"
        )
        if str(getattr(args, "controlled_target", "none")) != "none":
            controlled_target_actor, _ = _spawn_controlled_target(
                world=world,
                anchor_location=anchor_location,
                camera_transform=camera_transform,
                kind=str(args.controlled_target),
                speed_mps=float(args.target_speed_mps),
                fwd_dist_m=float(args.target_fwd_dist_m),
                span_m=float(args.target_span_m),
                vehicle_filter=str(args.target_vehicle_filter),
            )
            actors.append(controlled_target_actor)
        elif not _skip_background:
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
                # Control NPC traffic speed regime + light-stopping for the opportunity-window speed sweep.
                _npc_spd = getattr(args, "npc_speed_difference_pct", None)
                _npc_ign = float(getattr(args, "npc_ignore_lights_pct", 0.0) or 0.0)
                for _v in background_vehicles:
                    try:
                        if _npc_spd is not None:
                            traffic_manager.vehicle_percentage_speed_difference(_v, float(_npc_spd))
                        if _npc_ign > 0.0:
                            traffic_manager.ignore_lights_percentage(_v, max(0.0, min(100.0, _npc_ign)))
                    except Exception:
                        pass
                if _npc_spd is not None or _npc_ign > 0.0:
                    print(f"NPC traffic: speed_diff={_npc_spd}%  ignore_lights={_npc_ign:.0f}%")

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
            attach_to=camera_attach_to,
        )
        actors.append(camera)
        if str(getattr(args, "overlay_save_dir", "")):
            import os as _os
            _os.makedirs(args.overlay_save_dir, exist_ok=True)
            _sanity = {"n": 0}
            _every = max(1, int(args.overlay_save_every))
            def _cam_cb(image, _q=image_queue, _st=_sanity, _dir=args.overlay_save_dir, _ev=_every):
                od_demo.put_latest(_q, image)
                _st["n"] += 1
                if _st["n"] % _ev == 0:
                    try:
                        image.save_to_disk(_os.path.join(_dir, f"rgb_{image.frame:06d}.png"))
                    except Exception:
                        pass
            camera.listen(_cam_cb)
        else:
            camera.listen(lambda image: od_demo.put_latest(image_queue, image))

        if bool(args.enable_semantic_gt):
            gt_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
            gt_bp.set_attribute("image_size_x", str(int(camera_width)))
            gt_bp.set_attribute("image_size_y", str(int(camera_height)))
            gt_bp.set_attribute("fov", str(float(args.camera_fov)))
            gt_bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))
            gt_camera = world.spawn_actor(gt_bp, camera_transform, attach_to=camera_attach_to)
            actors.append(gt_camera)
            gt_queue = queue.Queue(maxsize=2)
            gt_camera.listen(lambda image, q=gt_queue: od_demo.put_latest(q, image))
            print("Semantic-GT camera enabled for per-frame fusion mask IoU.")

        radar_pipeline = PoleRadarPipeline(
            world=world,
            transform=radar_transform,
            attach_to=radar_attach_to,
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
        sensor_label = "Pole" if sensor_platform == "pole" else "Parked ego"
        print(f"{sensor_label} RGB camera ready on frame {first_image.frame}.")

        intrinsics_input = intrinsics_at(
            int(model_input_size[0]), int(model_input_size[1]), float(args.camera_fov)
        )
        intrinsics_display = intrinsics_at(
            int(camera_width),
            int(camera_height),
            float(args.camera_fov),
        )

        if str(args.spatial_map_stream_id).strip():
            spatial_stream_id = str(args.spatial_map_stream_id).strip()
        elif sensor_platform == "pole" and traffic_light is not None:
            spatial_stream_id = f"fusion_tl_{traffic_light.id}"
        else:
            spatial_stream_id = f"fusion_ego_{anchor_actor.id}"
        transport_label = _default_transport_label(args)

        if bool(args.run_logging):
            metrics_logger = FusionRunLogger.from_args(
                args=args,
                stream_id=spatial_stream_id,
                transport_label=transport_label,
            )
            metrics_logger.write_manifest(
                world=world,
                anchor_actor=anchor_actor,
                sensor_placement=(
                    "traffic_light_pole" if sensor_platform == "pole" else "ego_vehicle_front"
                ),
                anchor_label=(
                    f"traffic_light_{traffic_light.id}"
                    if sensor_platform == "pole" and traffic_light is not None
                    else f"ego_vehicle_{anchor_actor.id}"
                ),
                model_input_size=model_input_size,
                camera_width=int(camera_width),
                camera_height=int(camera_height),
                front_device=front_device,
                back_device=back_device,
                checkpoint_path=checkpoint_path,
                tracked_lead_actor=tracked_lead_actor,
                experiment3_target_actor=experiment3_target_actor,
                camera_actor=camera,
                radar_actor=radar_pipeline.sensor,
            )
            print(f"[Metrics] Run directory: {metrics_logger.run_dir}")
            print(f"[Metrics] Run group: {metrics_logger.run_group}")
            print(f"[Metrics] Stream CSV: {metrics_logger.csv_path}")

        if bool(args.spatial_map_stream):
            spatial_publisher = SpatialMapResultPublisher(
                host=str(args.spatial_map_host),
                port=int(args.spatial_map_port),
                stream_id=spatial_stream_id,
                traffic_light_id=(
                    str(args.traffic_light_id)
                    if sensor_platform == "pole"
                    else f"ego_vehicle_{anchor_actor.id}"
                ),
                traffic_light_actor_id=int(anchor_actor.id),
                traffic_light_opendrive_id=od_id if sensor_platform == "pole" else "",
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
        experiment3_cycle_frame_index = 0
        max_measurement_frames = max(0, int(args.max_frames))
        run_duration_s = max(0.0, float(args.run_duration_s))

        def run_duration_elapsed() -> bool:
            if run_duration_s <= 0.0:
                return False
            elapsed = time.perf_counter() - start_perf
            if elapsed < run_duration_s:
                return False
            print(f"Reached --run-duration-s={run_duration_s:.1f}; stopping run.")
            return True

        while True:
            if run_duration_elapsed():
                break
            if bool(args.sync_world):
                if (
                    experiment3_profile == "lateral_cycle"
                    and experiment3_target_actor is not None
                    and ego_vehicle is not None
                ):
                    _place_experiment3_target(
                        world=world,
                        ego_vehicle=ego_vehicle,
                        target_actor=experiment3_target_actor,
                        forward_m=float(args.experiment3_target_forward_m),
                        lateral_m=_experiment3_cycle_lateral_offset(
                            args,
                            experiment3_cycle_frame_index,
                        ),
                    )
                    experiment3_cycle_frame_index += 1
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
                if run_duration_elapsed():
                    break
                continue

            gt_3class: Optional[np.ndarray] = None
            if gt_queue is not None:
                if bool(args.sync_world):
                    gt_image = od_demo.wait_for_camera_frame(
                        gt_queue,
                        int(image.frame),
                        float(args.camera_timeout),
                    )
                else:
                    try:
                        gt_image = gt_queue.get(timeout=0.01)
                    except queue.Empty:
                        gt_image = None
                if gt_image is not None:
                    gt_tags = trained_seg_demo.carla_semantic_image_to_tags(gt_image)
                    gt_3class = trained_seg_demo.map_carla_tags_to_3class(gt_tags)

            radar_measurement = radar_pipeline.get_latest(timeout=float(args.camera_timeout))
            if radar_measurement is None:
                print(
                    f"Warning: radar measurement not received within {args.camera_timeout:.1f}s; "
                    "skipping frame."
                )
                if run_duration_elapsed():
                    break
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
                experiment3_target_radar_points = _radar_points_inside_actor_box(
                    np.asarray(radar_points.get("world_xyz", np.zeros((0, 3))), dtype=np.float32),
                    experiment3_target_actor,
                )
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
                        gt_3class=gt_3class,
                        spatial_publisher=spatial_publisher,
                        camera_width=int(camera_width),
                        camera_height=int(camera_height),
                        model_input_size=model_input_size,
                        anchor_actor=anchor_actor,
                        tracked_lead_actor=tracked_lead_actor,
                        experiment3_target_actor=experiment3_target_actor,
                        experiment3_target_radar_points=experiment3_target_radar_points,
                    )
                )
                metrics_logger.append_object_predictions(
                    elapsed_s=elapsed_s,
                    frame_id=int(image.frame),
                    objects=objects,
                )
                metrics_logger.append_object_ground_truth(
                    build_vehicle_ground_truth_rows(
                        world=world,
                        frame_id=int(image.frame),
                        elapsed_s=elapsed_s,
                        carla_timestamp=float(image.timestamp),
                        camera_transform=camera.get_transform(),
                        camera_inverse_matrix=camera_inverse_matrix,
                        intrinsics=intrinsics_display,
                        camera_width=int(camera_width),
                        camera_height=int(camera_height),
                        exclude_actor_ids=(
                            [int(anchor_actor.id)] if sensor_platform == "ego_vehicle" else []
                        ),
                    )
                )
            if max_measurement_frames > 0 and processed_frames >= max_measurement_frames:
                print(f"Reached --max-frames={max_measurement_frames}; stopping run.")
                break
            if run_duration_elapsed():
                break

            save_annotated = bool(str(getattr(args, "overlay_save_dir", ""))) and (
                processed_frames % max(1, int(args.overlay_save_every)) == 0
            )
            if gui_enabled or save_annotated:
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
                    traffic_light_id=(
                        str(args.traffic_light_id)
                        if sensor_platform == "pole"
                        else f"ego vehicle {anchor_actor.id}"
                    ),
                )
                if save_annotated:
                    cv2.imwrite(
                        str(
                            Path(args.overlay_save_dir)
                            / f"annotated_{int(image.frame):06d}.jpg"
                        ),
                        annotated,
                    )
                if gui_enabled:
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

        if exact_tracked_convoy:
            for convoy_actor in (ego_vehicle, tracked_lead_actor):
                if convoy_actor is None:
                    continue
                try:
                    convoy_actor.disable_constant_velocity()
                except RuntimeError:
                    pass

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

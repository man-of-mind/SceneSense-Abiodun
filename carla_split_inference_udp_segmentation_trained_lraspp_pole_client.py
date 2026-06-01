#!/usr/bin/env python3

"""
Traffic-light-pole RGB camera client for trained split semantic segmentation.

This hybrid client keeps the traffic-light-pole camera behavior from
carla_split_inference_udp_segmentation_pole_client.py, while using the trained
LR-ASPP checkpoint loading and 3-class CARLA semantic metrics path from
carla_split_inference_udp_segmentation_trained_lraspp_demo.py.

Press q or Esc in the OpenCV view to exit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import carla_split_inference_udp_demo as od_demo
import carla_split_inference_udp_data_collect as od_collect
import carla_split_inference_udp_segmentation_demo as seg_demo
import carla_split_inference_udp_segmentation_trained_lraspp_demo as trained_seg_demo

carla = trained_seg_demo.carla
cv2 = trained_seg_demo.cv2

DEFAULT_TRAFFIC_LIGHT_ID = "14"
DEFAULT_CAMERA_YAW_OFFSET_DEG = 90.0
DEFAULT_CAMERA_PITCH_DEG = -35.0
DEFAULT_SPAWN_RADIUS_METERS = 90.0
DEFAULT_WINDOW_NAME = "CARLA Pole Split Segmentation"
DEFAULT_LIVE_PLOT_REFRESH_SECONDS = od_demo.DEFAULT_LIVE_PLOT_REFRESH_SECONDS
DEFAULT_DETECTION_LOG_DIR = Path(__file__).resolve().parent / "metrics_logs" / "pole_segmentation_detections"
DEFAULT_METRICS_LOG_DIR = Path(__file__).resolve().parent / "metrics_logs" / "trained_pole_segmentation_metrics"
WEATHER_PRESET_NONE = trained_seg_demo.WEATHER_PRESET_NONE

VEHICLE_VOC_LABELS = {
    1,   # aeroplane, treated as vehicle for VOC compatibility
    2,   # bicycle
    4,   # boat
    6,   # bus
    7,   # car
    14,  # motorbike
    19,  # train
}
PERSON_VOC_LABELS = {15}
VOC_DETECTION_GROUPS: Tuple[Tuple[str, Sequence[int], Tuple[int, int, int]], ...] = (
    ("vehicle", tuple(sorted(VEHICLE_VOC_LABELS)), (0, 240, 255)),
    ("person", tuple(sorted(PERSON_VOC_LABELS)), (255, 255, 0)),
)
CARLA_3CLASS_DETECTION_GROUPS: Tuple[Tuple[str, Sequence[int], Tuple[int, int, int]], ...] = (
    ("vehicle", (trained_seg_demo.CLASS_ID_VEHICLE,), (0, 240, 255)),
    ("person", (trained_seg_demo.CLASS_ID_PERSON,), (255, 255, 0)),
)
DETECTION_LOG_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "frame_id",
    "row_type",
    "label",
    "detection_index",
    "gt_actor_id",
    "gt_source",
    "gt_actor_type_id",
    "match_iou",
    "pred_bbox_x",
    "pred_bbox_y",
    "pred_bbox_w",
    "pred_bbox_h",
    "pred_bbox_area_px",
    "pred_mask_area_px",
    "pred_center_x",
    "pred_center_y",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_w",
    "gt_bbox_h",
    "gt_bbox_area_px",
    "gt_center_x",
    "gt_center_y",
    "gt_depth_m",
    "gt_distance_m",
    "gt_extent_x_m",
    "gt_extent_y_m",
    "gt_extent_z_m",
    "gt_size_x_m",
    "gt_size_y_m",
    "gt_size_z_m",
    "pred_mask_to_gt_area_ratio",
    "pred_bbox_to_gt_area_ratio",
    "pred_to_gt_width_ratio",
    "pred_to_gt_height_ratio",
)


def detection_groups_for_args(
    args: argparse.Namespace,
) -> Tuple[Tuple[str, Sequence[int], Tuple[int, int, int]], ...]:
    if str(getattr(args, "seg_class_scheme", "carla_3class")) == "carla_3class":
        return CARLA_3CLASS_DETECTION_GROUPS
    return VOC_DETECTION_GROUPS


def mask_class_summary(mask: np.ndarray, args: argparse.Namespace, max_items: int = 4) -> str:
    labels, counts = np.unique(mask, return_counts=True)
    foreground = sorted(
        ((int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0),
        key=lambda item: item[1],
        reverse=True,
    )
    if not foreground:
        return "classes: background"
    label_names = trained_seg_demo.segmentation_label_names(args)
    parts = []
    for label, count in foreground[:max_items]:
        name = label_names[label] if label < len(label_names) else str(label)
        parts.append(f"{name} {count / mask.size * 100.0:.1f}%")
    return "classes: " + ", ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run split semantic segmentation over a traffic-light-mounted "
            "CARLA RGB camera using localhost UDP between model halves."
        )
    )
    parser.add_argument("--metrics-plot-worker", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--town",
        default="",
        help=(
            "Optional CARLA town to load before spawning the pole camera. "
            "Leave blank to use the currently loaded world."
        ),
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
        help="Traffic Manager port for optional background traffic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for optional traffic and pedestrian spawning.",
    )

    parser.add_argument(
        "--traffic-light-id",
        default=DEFAULT_TRAFFIC_LIGHT_ID,
        help=(
            "Traffic-light actor id to mount near. The selector also accepts "
            "a traffic light OpenDRIVE id when CARLA exposes one."
        ),
    )
    parser.add_argument(
        "--list-traffic-lights",
        action="store_true",
        help="List available traffic light ids and exit.",
    )
    parser.add_argument(
        "--camera-location-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "Interpret --camera-x/y/z as a traffic-light-local offset "
            "or as an absolute CARLA world location."
        ),
    )
    parser.add_argument(
        "--camera-x",
        type=float,
        default=0.0,
        help="Camera x offset from the selected pole, or world x in absolute mode.",
    )
    parser.add_argument(
        "--camera-y",
        type=float,
        default=0.0,
        help="Camera y offset from the selected pole, or world y in absolute mode.",
    )
    parser.add_argument(
        "--camera-z",
        type=float,
        default=5.0,
        help="Camera z offset above the selected pole, or world z in absolute mode.",
    )
    parser.add_argument(
        "--camera-yaw",
        type=float,
        default=None,
        help=(
            "Absolute camera yaw in degrees. Omit to use the selected traffic "
            "light yaw plus --camera-yaw-offset."
        ),
    )
    parser.add_argument(
        "--camera-yaw-offset",
        type=float,
        default=DEFAULT_CAMERA_YAW_OFFSET_DEG,
        help="Yaw offset from the traffic light yaw when --camera-yaw is omitted.",
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=DEFAULT_CAMERA_PITCH_DEG,
        help="Camera pitch in degrees. Negative values look downward.",
    )
    parser.add_argument("--camera-roll", type=float, default=0.0, help="Camera roll in degrees.")
    parser.add_argument("--camera-fov", type=float, default=90.0, help="RGB camera FoV in degrees.")
    parser.add_argument(
        "--camera-resolution",
        choices=["custom", *od_demo.CAMERA_RESOLUTION_PRESETS.keys()],
        default="custom",
        help="Preset camera resolution. Use custom to honor --camera-width/height.",
    )
    parser.add_argument("--camera-width", type=int, default=640, help="Custom camera width.")
    parser.add_argument("--camera-height", type=int, default=384, help="Custom camera height.")
    parser.add_argument("--fps", type=float, default=10.0, help="Synchronous sensor tick rate.")
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a camera frame before retrying.",
    )
    parser.add_argument(
        "--camera-warmup-ticks",
        type=int,
        default=8,
        help="Maximum synchronous ticks used to wait for the first camera frame.",
    )

    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync-world",
        dest="sync_world",
        action="store_true",
        help="Run the CARLA world in synchronous mode while this client is active.",
    )
    sync_group.add_argument(
        "--async-world",
        dest="sync_world",
        action="store_false",
        help="Do not force CARLA synchronous mode.",
    )
    parser.set_defaults(sync_world=True)

    parser.add_argument(
        "--npc-vehicles",
        type=int,
        default=20,
        help="Number of optional background autopilot vehicles to spawn near the pole.",
    )
    parser.add_argument(
        "--npc-pedestrians",
        type=int,
        default=30,
        help="Number of optional background pedestrians to spawn near the pole.",
    )
    parser.add_argument(
        "--spawn-radius",
        type=float,
        default=DEFAULT_SPAWN_RADIUS_METERS,
        help="Preferred radius, in meters, for optional traffic/pedestrian spawning.",
    )

    parser.add_argument(
        "--segmentation-model",
        choices=trained_seg_demo.SEGMENTATION_MODEL_CHOICES,
        default=trained_seg_demo.SEGMENTATION_MODEL_LRASPP,
        help="Semantic segmentation model used for split inference.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--seg-pretrained",
        dest="seg_pretrained",
        action="store_true",
        help="Use torchvision pretrained weights for the segmentation model.",
    )
    pretrained_group.add_argument(
        "--seg-disable-pretrained",
        dest="seg_pretrained",
        action="store_false",
        help="Do not load torchvision pretrained segmentation weights.",
    )
    parser.set_defaults(seg_pretrained=False)
    parser.add_argument(
        "--seg-weights-path",
        default="",
        help=(
            "Optional state_dict checkpoint for the segmentation model. "
            "For pole_lraspp_training, pass checkpoints/<trial>/best.pt."
        ),
    )
    parser.add_argument(
        "--trained-experiment-dir",
        default="",
        help=(
            "Optional pole_lraspp_training experiment directory. If provided "
            "and --seg-weights-path is omitted, the script reads manifest.json "
            "or checkpoint trial summaries to find the best checkpoint."
        ),
    )
    parser.add_argument(
        "--seg-num-classes",
        type=int,
        default=3,
        help="Number of output segmentation classes. The fine-tuned CARLA model uses 3.",
    )
    parser.add_argument(
        "--seg-class-scheme",
        choices=("carla_3class", "voc"),
        default="carla_3class",
        help="Prediction label scheme. Use carla_3class for pole_lraspp_training checkpoints.",
    )
    parser.add_argument(
        "--use-checkpoint-input-size",
        dest="use_checkpoint_input_size",
        action="store_true",
        help="Use checkpoint['input_size'] when present.",
    )
    parser.add_argument(
        "--disable-checkpoint-input-size",
        dest="use_checkpoint_input_size",
        action="store_false",
        help="Honor --seg-input-width/--seg-input-height even if the checkpoint stores an input size.",
    )
    parser.set_defaults(use_checkpoint_input_size=True)
    parser.add_argument(
        "--seg-input-width",
        type=int,
        default=512,
        help="Model input width. Use 0 to match the CARLA RGB camera width.",
    )
    parser.add_argument(
        "--seg-input-height",
        type=int,
        default=288,
        help="Model input height. Use 0 to match the CARLA RGB camera height.",
    )
    parser.add_argument(
        "--seg-mask-strength",
        type=float,
        default=0.78,
        help="Visualization overlay strength for predicted segmentation masks.",
    )
    parser.add_argument(
        "--min-detection-area",
        type=int,
        default=120,
        help="Minimum connected-component area, in pixels, counted as a detection.",
    )
    detection_log_group = parser.add_mutually_exclusive_group()
    detection_log_group.add_argument(
        "--enable-detection-log",
        dest="detection_log",
        action="store_true",
        help=(
            "Log per-frame segmentation component sizes and projected CARLA "
            "ground-truth bounding boxes to CSV."
        ),
    )
    detection_log_group.add_argument(
        "--disable-detection-log",
        dest="detection_log",
        action="store_false",
        help="Disable per-object segmentation/ground-truth CSV logging.",
    )
    parser.set_defaults(detection_log=True)
    parser.add_argument(
        "--detection-log-dir",
        default=str(DEFAULT_DETECTION_LOG_DIR),
        help="Directory where per-object segmentation/ground-truth CSV logs are saved.",
    )
    parser.add_argument(
        "--detection-log-prefix",
        default="pole_segmentation_object_sizes",
        help="Filename prefix for the per-object size CSV.",
    )
    parser.add_argument(
        "--detection-log-queue-size",
        type=int,
        default=4096,
        help="Maximum queued object-size log rows before old rows are dropped.",
    )
    parser.add_argument(
        "--detection-log-batch-size",
        type=int,
        default=200,
        help="Number of object-size rows written per CSV batch.",
    )
    parser.add_argument(
        "--detection-log-flush-interval",
        type=float,
        default=1.0,
        help="Maximum seconds between object-size CSV flushes.",
    )
    parser.add_argument(
        "--gt-match-iou-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum 2D IoU used to match a segmentation component to a CARLA "
            "ground-truth projected bounding box."
        ),
    )
    parser.add_argument(
        "--gt-max-distance",
        type=float,
        default=120.0,
        help="Ignore ground-truth actors/static boxes farther than this many meters from the camera.",
    )
    static_gt_group = parser.add_mutually_exclusive_group()
    static_gt_group.add_argument(
        "--include-static-level-bboxes",
        dest="include_static_level_bboxes",
        action="store_true",
        help=(
            "Also project static CARLA level bounding boxes, when available. "
            "This is useful for parked vehicles that are map geometry rather than actors."
        ),
    )
    static_gt_group.add_argument(
        "--disable-static-level-bboxes",
        dest="include_static_level_bboxes",
        action="store_false",
        help="Only use dynamic vehicle/walker actors for CARLA ground-truth boxes.",
    )
    parser.set_defaults(include_static_level_bboxes=True)
    parser.add_argument(
        "--postprocess-detection-log",
        default="",
        help="Generate offline overlay-size variation figures from a detection CSV and exit.",
    )
    parser.add_argument(
        "--postprocess-output-dir",
        default="",
        help=(
            "Directory for offline detection-log figures. Defaults to a figures "
            "subdirectory beside the CSV."
        ),
    )
    parser.add_argument(
        "--postprocess-max-actors",
        type=int,
        default=6,
        help="Maximum matched CARLA objects shown in per-object postprocess plots.",
    )

    parser.add_argument(
        "--metrics-log-dir",
        default=str(DEFAULT_METRICS_LOG_DIR),
        help="Directory where latency/payload/mIoU metrics CSV files are saved.",
    )
    parser.add_argument(
        "--metrics-log-prefix",
        default="trained_pole_split_segmentation_metrics",
        help="Filename prefix used for the metrics CSV.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional short tag stamped into every metrics CSV row and manifest.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, stop after this many post-warmup measurement frames.",
    )
    parser.add_argument(
        "--run-duration-s",
        type=float,
        default=0.0,
        help="If > 0, stop after this many post-warmup measurement seconds.",
    )
    parser.add_argument(
        "--weather-preset",
        default=WEATHER_PRESET_NONE,
        help=(
            "Optional carla.WeatherParameters preset name to apply at startup. "
            f"Default is '{WEATHER_PRESET_NONE}', which leaves weather unchanged."
        ),
    )
    parser.add_argument(
        "--manifest-extra-json",
        default="",
        help="Optional JSON object merged into the metrics run manifest.",
    )
    metrics_collection_group = parser.add_mutually_exclusive_group()
    metrics_collection_group.add_argument(
        "--enable-data-collection",
        "--enable-metrics-collection",
        dest="collect_metrics",
        action="store_true",
        help="Enable latency/payload/mIoU metrics CSV logging.",
    )
    metrics_collection_group.add_argument(
        "--disable-data-collection",
        "--disable-metrics-collection",
        dest="collect_metrics",
        action="store_false",
        help="Disable latency/payload/mIoU metrics CSV logging.",
    )
    parser.set_defaults(collect_metrics=True)
    parser.add_argument(
        "--metrics-warmup-frames",
        type=int,
        default=od_demo.DEFAULT_METRICS_WARMUP_FRAMES,
        help="Number of initial frames excluded while feature range trackers stabilize.",
    )
    parser.add_argument(
        "--metrics-batch-size",
        type=int,
        default=32,
        help="Number of queued metrics samples written to CSV per batch flush.",
    )
    parser.add_argument(
        "--metrics-flush-interval",
        type=float,
        default=1.0,
        help="Maximum seconds between background metrics CSV flushes.",
    )
    parser.add_argument(
        "--metrics-queue-size",
        type=int,
        default=1024,
        help="Maximum number of queued metrics samples before old samples are dropped.",
    )
    semantic_gt_group = parser.add_mutually_exclusive_group()
    semantic_gt_group.add_argument(
        "--enable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_true",
        help=(
            "Spawn a co-located CARLA semantic-segmentation camera and log "
            "3-class mIoU against CARLA ground truth."
        ),
    )
    semantic_gt_group.add_argument(
        "--disable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_false",
        help="Skip the semantic GT camera and mIoU metrics.",
    )
    parser.set_defaults(enable_semantic_gt=True)

    parser.add_argument(
        "--camera-source-port",
        type=int,
        default=37000,
        help="Local UDP source port used by the camera-side sender.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=37001,
        help="Local UDP receive port for the remote inference side.",
    )
    parser.add_argument(
        "--remote-source-port",
        type=int,
        default=37002,
        help="Local UDP source port used by the remote side sender.",
    )
    parser.add_argument(
        "--camera-result-port",
        type=int,
        default=37003,
        help="Local UDP receive port used by the camera side for masks.",
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
        help="Seconds to wait for a matching segmentation result.",
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

    live_plot_group = parser.add_mutually_exclusive_group()
    live_plot_group.add_argument(
        "--enable-live-plot",
        dest="live_plot",
        action="store_true",
        help="Enable the real-time matplotlib latency/count plot window.",
    )
    live_plot_group.add_argument(
        "--disable-live-plot",
        dest="live_plot",
        action="store_false",
        help="Disable the real-time matplotlib latency/count plot window.",
    )
    parser.set_defaults(live_plot=True)
    parser.add_argument(
        "--live-plot-history",
        type=int,
        default=300,
        help="Number of recent samples shown in the live metrics plot.",
    )
    parser.add_argument(
        "--live-plot-update-interval",
        type=int,
        default=5,
        help="Send one live-plot update every N processed frames.",
    )
    parser.add_argument(
        "--live-plot-refresh-seconds",
        type=float,
        default=DEFAULT_LIVE_PLOT_REFRESH_SECONDS,
        help="Seconds between GUI refreshes inside the live plot worker.",
    )
    parser.add_argument(
        "--live-plot-queue-size",
        type=int,
        default=512,
        help="Maximum queued live-plot samples before old samples are dropped.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the OpenCV camera view and matplotlib live plot.",
    )

    parser.add_argument(
        "--quantization-mode",
        choices=od_collect.QUANT_MODE_CHOICES,
        default=od_collect.QUANT_MODE_PER_TENSOR_UINT8,
        help="How feature tensors are quantized before UDP serialization.",
    )
    parser.add_argument(
        "--entropy-coder",
        choices=od_collect.ENTROPY_CODER_CHOICES,
        default=od_collect.ENTROPY_CODER_ZLIB,
        help="Entropy coder applied to the pickled UDP payload.",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        help="zstd compression level when --entropy-coder=zstd.",
    )
    parser.add_argument(
        "--roi-objectness-threshold",
        type=float,
        default=0.0,
        help=(
            "Segmentation saliency-gate target drop fraction in [0,1). "
            "0 disables the gate."
        ),
    )
    parser.add_argument(
        "--ae-mode",
        choices=od_collect.AE_MODE_CHOICES,
        default=od_collect.AE_MODE_OFF,
        help="Optional per-level feature autoencoder bottleneck before UDP transport.",
    )
    parser.add_argument(
        "--ae-bottleneck-channels",
        type=int,
        default=64,
        help="Channel count at the autoencoder bottleneck, per backbone level.",
    )
    parser.add_argument(
        "--ae-spatial-stride",
        type=int,
        default=1,
        help="Spatial stride applied by the autoencoder encoder.",
    )
    parser.add_argument(
        "--ae-checkpoint",
        default="",
        help="Path to a torch.save() blob containing per-level autoencoder state dicts.",
    )
    parser.add_argument(
        "--ae-seed",
        type=int,
        default=0,
        help="Seed used when --ae-mode=random_projection so both sides match.",
    )
    parser.add_argument(
        "--per-level-compress-probe",
        action="store_true",
        help="Collect per-level compression byte counts internally while sending features.",
    )
    parser.set_defaults(per_level_compress_probe=True)
    return parser.parse_args()


def _record_float(record: Dict[str, object], key: str) -> float:
    value = record.get(key)
    if value is None:
        return float("nan")
    return float(value)


def render_live_metrics_axes(
    axes: Tuple[object, object, object],
    records: Sequence[Dict[str, object]],
) -> None:
    latency_ax, payload_ax, count_ax = axes
    for axis in axes:
        axis.clear()
        axis.grid(True, alpha=0.3)

    latency_ax.set_title("Traffic-Light Split Segmentation Metrics")
    latency_ax.set_ylabel("Latency (ms)")
    payload_ax.set_ylabel("UDP payload (KiB)")
    count_ax.set_ylabel("Detections")
    count_ax.set_xlabel("Elapsed time (s)")
    if not records:
        return

    elapsed = [_record_float(record, "elapsed_s") for record in records]
    front_ms = [_record_float(record, "front_ms") for record in records]
    back_ms = [_record_float(record, "back_ms") for record in records]
    round_trip_ms = [_record_float(record, "round_trip_ms") for record in records]
    payload_kib = [_record_float(record, "payload_kib") for record in records]
    payload_uncompressed_kib = [
        _record_float(record, "payload_uncompressed_kib") for record in records
    ]
    detections = [_record_float(record, "detections") for record in records]

    latency_ax.plot(elapsed, front_ms, label="Front half", color="tab:blue", linewidth=1.8)
    latency_ax.plot(elapsed, back_ms, label="Back half", color="tab:orange", linewidth=1.8)
    latency_ax.plot(elapsed, round_trip_ms, label="Round trip", color="tab:red", linewidth=1.8)
    latency_ax.legend(loc="upper right")
    payload_ax.plot(
        elapsed,
        payload_kib,
        label="Compressed UDP payload",
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
    count_ax.step(elapsed, detections, where="post", color="tab:green", linewidth=1.8)
    count_ax.set_ylim(bottom=0)


class LiveMetricsPlotter:
    def __init__(self, history_limit: int, refresh_seconds: float) -> None:
        self.history_limit = max(10, int(history_limit))
        self.refresh_seconds = max(0.05, float(refresh_seconds))
        self.records: Deque[Dict[str, object]] = deque(maxlen=self.history_limit)
        self.enabled = False
        self.warning: Optional[str] = None
        try:
            import matplotlib

            matplotlib.use("TkAgg", force=True)
            import matplotlib.pyplot as plt

            self._plt = plt
            self._plt.ion()
            self._figure, axes = self._plt.subplots(
                3,
                1,
                figsize=(10, 8),
                sharex=True,
                constrained_layout=True,
            )
            self._axes = tuple(axes)
            try:
                self._figure.canvas.manager.set_window_title(
                    "CARLA Pole Split Segmentation Metrics"
                )
            except Exception:
                pass
            render_live_metrics_axes(self._axes, list(self.records))
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
        render_live_metrics_axes(self._axes, list(self.records))
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
        refresh_seconds=args.live_plot_refresh_seconds,
    )
    if plotter.warning:
        print(plotter.warning, file=sys.stderr)
    if not plotter.enabled:
        return 1

    try:
        while plotter.enabled:
            ready, _, _ = select.select([sys.stdin], [], [], plotter.refresh_seconds)
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


class AsyncLivePlotSender(threading.Thread):
    def __init__(self, args: argparse.Namespace, gui_enabled: bool) -> None:
        super().__init__(daemon=True)
        self.queue: "queue.Queue[Optional[Dict[str, object]]]" = queue.Queue(
            maxsize=max(32, int(args.live_plot_queue_size))
        )
        self.warning: Optional[str] = None
        self._stopped = threading.Event()
        self._dropped_samples = 0
        self.process: Optional[subprocess.Popen[str]] = None
        self.stdin = None

        if not bool(args.live_plot):
            return
        if not gui_enabled:
            self.warning = "Live metrics plot disabled: running without a graphical display."
            return

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "carla_pole_segmentation_mpl"))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--metrics-plot-worker",
            "--live-plot-history",
            str(args.live_plot_history),
            "--live-plot-refresh-seconds",
            str(args.live_plot_refresh_seconds),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
            self.stdin = self.process.stdin
        except Exception as exc:
            self.warning = f"Live metrics plot disabled: unable to start worker ({exc})"
            self.process = None
            self.stdin = None

    def enabled(self) -> bool:
        return self.stdin is not None

    def submit(self, record: Dict[str, object]) -> None:
        if self._stopped.is_set() or self.stdin is None:
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
        if self.ident is not None:
            self.join(timeout=3.0)
        if self.stdin is not None:
            try:
                self.stdin.close()
            except Exception:
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        if self._dropped_samples:
            print(
                f"Warning: dropped {self._dropped_samples} live-plot samples to preserve responsiveness.",
                file=sys.stderr,
            )

    def run(self) -> None:
        while True:
            try:
                record = self.queue.get(timeout=0.2)
            except queue.Empty:
                if self._stopped.is_set():
                    break
                continue
            if record is None:
                if self._stopped.is_set():
                    break
                continue
            if self.stdin is None:
                continue
            try:
                self.stdin.write(json.dumps(record, allow_nan=True) + "\n")
                self.stdin.flush()
            except Exception:
                self.stdin = None
                self.process = None


def resolve_detection_log_path(args: argparse.Namespace) -> Path:
    output_dir = Path(args.detection_log_dir).expanduser().resolve()
    prefix = str(args.detection_log_prefix or "pole_segmentation_object_sizes").strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{timestamp}.csv"


class AsyncDetectionLogWriter(threading.Thread):
    def __init__(self, csv_path: Path, args: argparse.Namespace) -> None:
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue: "queue.Queue[Optional[List[Dict[str, object]]]]" = queue.Queue(
            maxsize=max(32, int(args.detection_log_queue_size))
        )
        self.batch_size = max(1, int(args.detection_log_batch_size))
        self.flush_interval = max(0.1, float(args.detection_log_flush_interval))
        self._stopped = threading.Event()
        self._dropped_rows = 0

    def submit_many(self, rows: Sequence[Dict[str, object]]) -> None:
        if self._stopped.is_set() or not rows:
            return
        rows_list = [dict(row) for row in rows]
        try:
            self.queue.put_nowait(rows_list)
            return
        except queue.Full:
            pass
        try:
            dropped = self.queue.get_nowait()
            if dropped:
                self._dropped_rows += len(dropped)
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(rows_list)
        except queue.Full:
            self._dropped_rows += len(rows_list)

    def close(self) -> None:
        self._stopped.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.ident is not None:
            self.join(timeout=5.0)
        if self._dropped_rows:
            print(
                f"Warning: dropped {self._dropped_rows} object-size log rows to preserve responsiveness.",
                file=sys.stderr,
            )

    def run(self) -> None:
        pending: List[Dict[str, object]] = []
        last_flush = time.monotonic()
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DETECTION_LOG_FIELDS)
            writer.writeheader()
            csv_file.flush()
            while True:
                timeout = max(0.05, self.flush_interval / 2.0)
                try:
                    rows = self.queue.get(timeout=timeout)
                except queue.Empty:
                    rows = None

                if rows is None:
                    if self._stopped.is_set():
                        break
                else:
                    pending.extend(rows)

                should_flush = (
                    pending
                    and (
                        len(pending) >= self.batch_size
                        or time.monotonic() - last_flush >= self.flush_interval
                        or self._stopped.is_set()
                    )
                )
                if should_flush:
                    writer.writerows(pending)
                    csv_file.flush()
                    pending.clear()
                    last_flush = time.monotonic()

            if pending:
                writer.writerows(pending)
                csv_file.flush()


def _traffic_light_id_candidates(actor: "carla.Actor") -> List[str]:
    candidates = [str(actor.id)]
    try:
        opendrive_id = actor.get_opendrive_id()
    except Exception:
        opendrive_id = None
    if opendrive_id not in (None, ""):
        candidates.append(str(opendrive_id))
    return candidates


def _traffic_light_opendrive_id(actor: "carla.Actor") -> str:
    try:
        value = actor.get_opendrive_id()
    except Exception:
        return ""
    return "" if value is None else str(value)


def list_traffic_lights(world: "carla.World") -> None:
    actors = sorted(world.get_actors().filter("traffic.traffic_light"), key=lambda actor: actor.id)
    print(f"Traffic lights in {world.get_map().name}: {len(actors)}")
    for actor in actors:
        transform = actor.get_transform()
        location = transform.location
        od_id = _traffic_light_opendrive_id(actor)
        od_text = f", opendrive_id={od_id}" if od_id else ""
        print(
            f"  id={actor.id}{od_text}, "
            f"loc=({location.x:.2f}, {location.y:.2f}, {location.z:.2f}), "
            f"yaw={transform.rotation.yaw:.1f}"
        )


def resolve_traffic_light(world: "carla.World", requested_id: str) -> "carla.Actor":
    requested = str(requested_id).strip()
    traffic_lights = list(world.get_actors().filter("traffic.traffic_light"))
    for actor in traffic_lights:
        if requested in _traffic_light_id_candidates(actor):
            return actor
    available = ", ".join(str(actor.id) for actor in sorted(traffic_lights, key=lambda item: item.id))
    raise ValueError(
        f"Traffic light id {requested!r} was not found in {world.get_map().name}. "
        f"Available actor ids: {available or 'none'}"
    )


def _transform_relative_location(
    base_transform: "carla.Transform",
    offset: "carla.Location",
) -> "carla.Location":
    matrix = np.array(base_transform.get_matrix(), dtype=np.float64)
    point = np.array([offset.x, offset.y, offset.z, 1.0], dtype=np.float64)
    x, y, z, _ = matrix @ point
    return carla.Location(x=float(x), y=float(y), z=float(z))


def build_camera_transform(
    traffic_light: "carla.Actor",
    args: argparse.Namespace,
) -> "carla.Transform":
    pole_transform = traffic_light.get_transform()
    local_or_world = carla.Location(
        x=float(args.camera_x),
        y=float(args.camera_y),
        z=float(args.camera_z),
    )
    if args.camera_location_mode == "relative":
        location = _transform_relative_location(pole_transform, local_or_world)
    else:
        location = local_or_world

    yaw = (
        float(args.camera_yaw)
        if args.camera_yaw is not None
        else float(pole_transform.rotation.yaw) + float(args.camera_yaw_offset)
    )
    rotation = carla.Rotation(
        pitch=float(args.camera_pitch),
        yaw=yaw,
        roll=float(args.camera_roll),
    )
    return carla.Transform(location, rotation)


def _distance_to(location: "carla.Location", other: "carla.Location") -> float:
    return float(location.distance(other))


def spawn_background_vehicles_near(
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    anchor_location: "carla.Location",
    count: int,
    radius: float,
) -> List["carla.Actor"]:
    if count <= 0:
        return []
    blueprints = od_demo.choose_vehicle_blueprints(world, cars_only=True)
    if not blueprints:
        print("No vehicle blueprints were found in the current CARLA world.")
        return []
    blueprint_ids = [blueprint.id for blueprint in blueprints]
    spawn_points = world.get_map().get_spawn_points()
    nearby = [
        point for point in spawn_points if _distance_to(point.location, anchor_location) <= radius
    ]
    candidates = nearby if nearby else spawn_points
    candidates = sorted(candidates, key=lambda point: _distance_to(point.location, anchor_location))
    candidates = candidates[: max(count * 3, count)]
    random.shuffle(candidates)

    batch = []
    for spawn_point in candidates[:count]:
        command = carla.command.SpawnActor(
            od_demo.get_fresh_vehicle_blueprint(
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

    spawned: List["carla.Actor"] = []
    if batch:
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                continue
            actor = world.get_actor(response.actor_id)
            if actor is not None:
                spawned.append(actor)
    if len(spawned) < count:
        print(f"Spawned {len(spawned)} background vehicles instead of requested {count}.")
    return spawned


def _choose_pedestrian_spawn_points_near(
    world: "carla.World",
    anchor_location: "carla.Location",
    count: int,
    radius: float,
) -> List["carla.Transform"]:
    spawn_points: List["carla.Transform"] = []
    attempts = max(count * 40, 80)
    for _ in range(attempts):
        if len(spawn_points) >= count:
            break
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        if _distance_to(location, anchor_location) > radius:
            continue
        spawn_points.append(
            carla.Transform(carla.Location(x=location.x, y=location.y, z=location.z + 1.0))
        )

    fallback_attempts = max(count * 20, 40)
    for _ in range(fallback_attempts):
        if len(spawn_points) >= count:
            break
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        spawn_points.append(
            carla.Transform(carla.Location(x=location.x, y=location.y, z=location.z + 1.0))
        )
    return spawn_points


def spawn_background_pedestrians_near(
    client: "carla.Client",
    world: "carla.World",
    anchor_location: "carla.Location",
    count: int,
    radius: float,
) -> Tuple[List["carla.Actor"], List["carla.Actor"]]:
    if count <= 0:
        return [], []
    pedestrian_blueprints = od_demo.choose_pedestrian_blueprints(world)
    if not pedestrian_blueprints:
        print("No pedestrian blueprints were found in the current CARLA world.")
        return [], []

    spawn_points = _choose_pedestrian_spawn_points_near(world, anchor_location, count, radius)
    if len(spawn_points) < count:
        print(f"Found {len(spawn_points)} pedestrian spawn locations instead of requested {count}.")

    walker_batch = []
    walker_speeds: List[float] = []
    for spawn_point in spawn_points:
        blueprint_id = random.choice(pedestrian_blueprints).id
        blueprint = od_demo.get_fresh_pedestrian_blueprint(world, blueprint_id)
        walker_batch.append(carla.command.SpawnActor(blueprint, spawn_point))
        walker_speeds.append(od_demo.resolve_pedestrian_speed(blueprint))

    walker_ids: List[int] = []
    walker_speeds_spawned: List[float] = []
    if walker_batch:
        for response, speed in zip(client.apply_batch_sync(walker_batch, True), walker_speeds):
            if response.error:
                continue
            walker_ids.append(response.actor_id)
            walker_speeds_spawned.append(speed)
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
    controller_speeds: List[float] = []
    for walker_id, speed, response in zip(
        walker_ids,
        walker_speeds_spawned,
        client.apply_batch_sync(controller_batch, True),
    ):
        if response.error:
            continue
        controller_ids.append(response.actor_id)
        controller_speeds.append(speed)

    walkers = [world.get_actor(actor_id) for actor_id in walker_ids]
    walkers = [actor for actor in walkers if actor is not None]
    controllers_with_speeds = []
    for actor_id, speed in zip(controller_ids, controller_speeds):
        actor = world.get_actor(actor_id)
        if actor is not None:
            controllers_with_speeds.append((actor, speed))
    controllers = [actor for actor, _ in controllers_with_speeds]

    if controllers:
        world.set_pedestrians_cross_factor(1.0)
        world.tick()
        for controller, speed in controllers_with_speeds:
            try:
                controller.start()
                destination = world.get_random_location_from_navigation()
                if destination is not None:
                    controller.go_to_location(destination)
                controller.set_max_speed(float(speed))
            except RuntimeError:
                continue
    return walkers, controllers


def summarize_segmentation_detections(
    mask: Optional[np.ndarray],
    min_area: int,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if mask is None:
        return []
    detections: List[Dict[str, object]] = []
    min_area = max(1, int(min_area))
    for label, class_ids, color_bgr in detection_groups_for_args(args):
        binary = np.isin(mask, class_ids).astype(np.uint8)
        if not binary.any():
            continue
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[component_index, cv2.CC_STAT_LEFT])
            y = int(stats[component_index, cv2.CC_STAT_TOP])
            width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            detections.append(
                {
                    "label": label,
                    "area": area,
                    "bbox": (x, y, width, height),
                    "bbox_area": int(width * height),
                    "center": (float(x + width / 2.0), float(y + height / 2.0)),
                    "color": color_bgr,
                }
            )
    detections.sort(key=lambda item: int(item["area"]), reverse=True)
    return detections


def _bbox_xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in bbox)
    return x, y, x + width, y + height


def _bbox_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_xywh_to_xyxy(b)
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
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _bbox_corner_offsets(extent: "carla.Vector3D") -> np.ndarray:
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


def _static_bbox_world_corners(bbox: "carla.BoundingBox") -> np.ndarray:
    rotation = getattr(bbox, "rotation", carla.Rotation())
    transform = carla.Transform(bbox.location, rotation)
    matrix = np.array(transform.get_matrix(), dtype=np.float32)
    corners_local = _bbox_corner_offsets(bbox.extent)
    homogeneous = np.concatenate(
        [corners_local, np.ones((corners_local.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (matrix @ homogeneous.T).T[:, :3]


def _project_corners_to_bbox(
    corners_world: np.ndarray,
    cam_inv_matrix: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Optional[Dict[str, float]]:
    if corners_world.size == 0:
        return None
    corners_cam = od_collect._world_to_camera_points(corners_world, cam_inv_matrix)
    u, v, depths = od_collect._project_camera_points_to_image(corners_cam, K)
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
        "bbox": (u_min, v_min, pixel_w, pixel_h),
        "x": u_min,
        "y": v_min,
        "w": pixel_w,
        "h": pixel_h,
        "area_px": pixel_w * pixel_h,
        "center_x": u_min + pixel_w / 2.0,
        "center_y": v_min + pixel_h / 2.0,
        "min_depth_m": float(np.min(depths)),
    }


def _actor_world_corners(actor: "carla.Actor") -> Optional[np.ndarray]:
    try:
        return od_collect._actor_bbox_world_corners(actor)
    except Exception:
        return None


def _project_actor_ground_truth(
    actor: "carla.Actor",
    label: str,
    camera_location: "carla.Location",
    cam_inv_matrix: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Optional[Dict[str, object]]:
    corners = _actor_world_corners(actor)
    if corners is None:
        return None
    projection = _project_corners_to_bbox(corners, cam_inv_matrix, K, width, height)
    if projection is None:
        return None
    bbox = getattr(actor, "bounding_box", None)
    extent = getattr(bbox, "extent", None) if bbox is not None else None
    extent_x = float(getattr(extent, "x", 0.0))
    extent_y = float(getattr(extent, "y", 0.0))
    extent_z = float(getattr(extent, "z", 0.0))
    try:
        distance_m = float(actor.get_location().distance(camera_location))
    except RuntimeError:
        distance_m = float("nan")
    return {
        "label": label,
        "gt_actor_id": str(actor.id),
        "gt_source": "actor",
        "gt_actor_type_id": str(actor.type_id),
        "gt_bbox": projection["bbox"],
        "gt_bbox_x": projection["x"],
        "gt_bbox_y": projection["y"],
        "gt_bbox_w": projection["w"],
        "gt_bbox_h": projection["h"],
        "gt_bbox_area_px": projection["area_px"],
        "gt_center_x": projection["center_x"],
        "gt_center_y": projection["center_y"],
        "gt_depth_m": projection["min_depth_m"],
        "gt_distance_m": distance_m,
        "gt_extent_x_m": extent_x,
        "gt_extent_y_m": extent_y,
        "gt_extent_z_m": extent_z,
        "gt_size_x_m": extent_x * 2.0,
        "gt_size_y_m": extent_y * 2.0,
        "gt_size_z_m": extent_z * 2.0,
    }


def _project_static_bbox_ground_truth(
    bbox: "carla.BoundingBox",
    *,
    label: str,
    source_index: int,
    source_name: str,
    camera_location: "carla.Location",
    cam_inv_matrix: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Optional[Dict[str, object]]:
    corners = _static_bbox_world_corners(bbox)
    projection = _project_corners_to_bbox(corners, cam_inv_matrix, K, width, height)
    if projection is None:
        return None
    extent = bbox.extent
    distance_m = float(bbox.location.distance(camera_location))
    return {
        "label": label,
        "gt_actor_id": f"{source_name}_{source_index}",
        "gt_source": source_name,
        "gt_actor_type_id": f"static.{label}",
        "gt_bbox": projection["bbox"],
        "gt_bbox_x": projection["x"],
        "gt_bbox_y": projection["y"],
        "gt_bbox_w": projection["w"],
        "gt_bbox_h": projection["h"],
        "gt_bbox_area_px": projection["area_px"],
        "gt_center_x": projection["center_x"],
        "gt_center_y": projection["center_y"],
        "gt_depth_m": projection["min_depth_m"],
        "gt_distance_m": distance_m,
        "gt_extent_x_m": float(extent.x),
        "gt_extent_y_m": float(extent.y),
        "gt_extent_z_m": float(extent.z),
        "gt_size_x_m": float(extent.x) * 2.0,
        "gt_size_y_m": float(extent.y) * 2.0,
        "gt_size_z_m": float(extent.z) * 2.0,
    }


def _iter_static_level_bboxes(world: "carla.World") -> Iterable[Tuple[str, str, "carla.BoundingBox"]]:
    labels = [("vehicle", "Vehicles")]
    pedestrian_label = getattr(carla.CityObjectLabel, "Pedestrians", None)
    if pedestrian_label is not None:
        labels.append(("person", "Pedestrians"))

    if hasattr(world, "get_level_bbs"):
        for label, carla_label_name in labels:
            carla_label = getattr(carla.CityObjectLabel, carla_label_name, None)
            if carla_label is None:
                continue
            try:
                for bbox in world.get_level_bbs(carla_label):
                    yield label, "level_bbox", bbox
            except Exception:
                continue
        return

    if hasattr(world, "get_environment_objects"):
        for label, carla_label_name in labels:
            carla_label = getattr(carla.CityObjectLabel, carla_label_name, None)
            if carla_label is None:
                continue
            try:
                for obj in world.get_environment_objects(carla_label):
                    bbox = getattr(obj, "bounding_box", None)
                    if bbox is not None:
                        yield label, "environment_object", bbox
            except Exception:
                continue


def project_ground_truth_objects(
    world: "carla.World",
    camera_actor: Optional["carla.Actor"],
    *,
    width: int,
    height: int,
    fov: float,
    max_distance_m: float,
    include_static_level_bboxes: bool,
    include_dynamic_actors: bool = True,
) -> List[Dict[str, object]]:
    if camera_actor is None:
        return []
    try:
        camera_transform = camera_actor.get_transform()
        camera_location = camera_transform.location
        cam_inv_matrix = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)
    except RuntimeError:
        return []
    K = od_collect.get_camera_intrinsics(int(width), int(height), float(fov))
    max_distance_m = max(0.0, float(max_distance_m))
    gt_objects: List[Dict[str, object]] = []

    if include_dynamic_actors:
        actors = world.get_actors()
        for label, filter_pattern in (("vehicle", "vehicle.*"), ("person", "walker.pedestrian.*")):
            for actor in actors.filter(filter_pattern):
                if actor.id == camera_actor.id:
                    continue
                try:
                    distance = float(actor.get_location().distance(camera_location))
                except RuntimeError:
                    continue
                if max_distance_m > 0.0 and distance > max_distance_m:
                    continue
                projected = _project_actor_ground_truth(
                    actor,
                    label,
                    camera_location,
                    cam_inv_matrix,
                    K,
                    int(width),
                    int(height),
                )
                if projected is not None:
                    gt_objects.append(projected)

    if include_static_level_bboxes:
        for source_index, (label, source_name, bbox) in enumerate(_iter_static_level_bboxes(world)):
            distance = float(bbox.location.distance(camera_location))
            if max_distance_m > 0.0 and distance > max_distance_m:
                continue
            projected = _project_static_bbox_ground_truth(
                bbox,
                label=label,
                source_index=source_index,
                source_name=source_name,
                camera_location=camera_location,
                cam_inv_matrix=cam_inv_matrix,
                K=K,
                width=int(width),
                height=int(height),
            )
            if projected is not None:
                gt_objects.append(projected)
    return gt_objects


def match_detections_to_ground_truth(
    detections: Sequence[Dict[str, object]],
    gt_objects: Sequence[Dict[str, object]],
    min_iou: float,
) -> Dict[int, Tuple[int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for det_index, detection in enumerate(detections):
        det_bbox = detection["bbox"]
        det_label = str(detection["label"])
        for gt_index, gt in enumerate(gt_objects):
            if det_label != str(gt["label"]):
                continue
            iou = _bbox_iou_xywh(det_bbox, gt["gt_bbox"])
            if iou >= float(min_iou):
                candidates.append((iou, det_index, gt_index))
    candidates.sort(reverse=True)

    matches: Dict[int, Tuple[int, float]] = {}
    used_detections = set()
    used_gt = set()
    for iou, det_index, gt_index in candidates:
        if det_index in used_detections or gt_index in used_gt:
            continue
        used_detections.add(det_index)
        used_gt.add(gt_index)
        matches[det_index] = (gt_index, float(iou))
    return matches


def _empty_detection_log_row() -> Dict[str, object]:
    return {field: "" for field in DETECTION_LOG_FIELDS}


def _fill_prediction_fields(row: Dict[str, object], detection: Dict[str, object]) -> None:
    x, y, width, height = detection["bbox"]
    center_x, center_y = detection["center"]
    row.update(
        {
            "label": str(detection["label"]),
            "pred_bbox_x": float(x),
            "pred_bbox_y": float(y),
            "pred_bbox_w": float(width),
            "pred_bbox_h": float(height),
            "pred_bbox_area_px": float(detection.get("bbox_area", width * height)),
            "pred_mask_area_px": float(detection["area"]),
            "pred_center_x": float(center_x),
            "pred_center_y": float(center_y),
        }
    )


def _fill_gt_fields(row: Dict[str, object], gt: Dict[str, object]) -> None:
    row.update(
        {
            "label": str(gt["label"]),
            "gt_actor_id": str(gt["gt_actor_id"]),
            "gt_source": str(gt["gt_source"]),
            "gt_actor_type_id": str(gt["gt_actor_type_id"]),
            "gt_bbox_x": float(gt["gt_bbox_x"]),
            "gt_bbox_y": float(gt["gt_bbox_y"]),
            "gt_bbox_w": float(gt["gt_bbox_w"]),
            "gt_bbox_h": float(gt["gt_bbox_h"]),
            "gt_bbox_area_px": float(gt["gt_bbox_area_px"]),
            "gt_center_x": float(gt["gt_center_x"]),
            "gt_center_y": float(gt["gt_center_y"]),
            "gt_depth_m": float(gt["gt_depth_m"]),
            "gt_distance_m": float(gt["gt_distance_m"]),
            "gt_extent_x_m": float(gt["gt_extent_x_m"]),
            "gt_extent_y_m": float(gt["gt_extent_y_m"]),
            "gt_extent_z_m": float(gt["gt_extent_z_m"]),
            "gt_size_x_m": float(gt["gt_size_x_m"]),
            "gt_size_y_m": float(gt["gt_size_y_m"]),
            "gt_size_z_m": float(gt["gt_size_z_m"]),
        }
    )


def build_detection_log_rows(
    *,
    frame_id: int,
    elapsed_s: float,
    detections: Sequence[Dict[str, object]],
    gt_objects: Sequence[Dict[str, object]],
    min_iou: float,
) -> List[Dict[str, object]]:
    now_iso = datetime.now().isoformat(timespec="milliseconds")
    common = {
        "wall_time_iso": now_iso,
        "elapsed_s": float(elapsed_s),
        "frame_id": int(frame_id),
    }
    matches = match_detections_to_ground_truth(detections, gt_objects, min_iou)
    matched_gt_indices = {gt_index for gt_index, _ in matches.values()}
    rows: List[Dict[str, object]] = []

    for det_index, detection in enumerate(detections):
        row = _empty_detection_log_row()
        row.update(common)
        row["row_type"] = "matched_detection" if det_index in matches else "unmatched_detection"
        row["detection_index"] = int(det_index)
        _fill_prediction_fields(row, detection)
        if det_index in matches:
            gt_index, iou = matches[det_index]
            gt = gt_objects[gt_index]
            _fill_gt_fields(row, gt)
            gt_area = max(float(gt["gt_bbox_area_px"]), 1.0)
            gt_width = max(float(gt["gt_bbox_w"]), 1.0)
            gt_height = max(float(gt["gt_bbox_h"]), 1.0)
            row.update(
                {
                    "match_iou": float(iou),
                    "pred_mask_to_gt_area_ratio": float(detection["area"]) / gt_area,
                    "pred_bbox_to_gt_area_ratio": float(detection["bbox_area"]) / gt_area,
                    "pred_to_gt_width_ratio": float(detection["bbox"][2]) / gt_width,
                    "pred_to_gt_height_ratio": float(detection["bbox"][3]) / gt_height,
                }
            )
        rows.append(row)

    for gt_index, gt in enumerate(gt_objects):
        if gt_index in matched_gt_indices:
            continue
        row = _empty_detection_log_row()
        row.update(common)
        row["row_type"] = "unmatched_gt"
        row["detection_index"] = -1
        _fill_gt_fields(row, gt)
        rows.append(row)

    return rows


def _save_postprocess_figure(figure: object, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base_path.with_suffix(".png"), dpi=150)
    figure.savefig(base_path.with_suffix(".pdf"))


def run_detection_log_postprocess(args: argparse.Namespace) -> int:
    csv_path = Path(args.postprocess_detection_log).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Detection log CSV not found: {csv_path}")
    output_dir = (
        Path(args.postprocess_output_dir).expanduser().resolve()
        if str(args.postprocess_output_dir or "").strip()
        else csv_path.parent / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"No rows found in {csv_path}")
        return 1

    for column in (
        "elapsed_s",
        "frame_id",
        "match_iou",
        "pred_bbox_area_px",
        "pred_mask_area_px",
        "gt_bbox_area_px",
        "pred_mask_to_gt_area_ratio",
        "pred_bbox_to_gt_area_ratio",
        "pred_to_gt_width_ratio",
        "pred_to_gt_height_ratio",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    matched = df[df["row_type"] == "matched_detection"].copy()
    matched = matched.dropna(subset=["gt_actor_id", "pred_mask_area_px", "gt_bbox_area_px"])
    if matched.empty:
        print(
            "No matched detection rows were found. "
            "Try lowering --gt-match-iou-threshold or checking the camera viewpoint."
        )
        return 1

    matched["gt_actor_key"] = (
        matched["gt_actor_id"].astype(str)
        + " | "
        + matched["label"].astype(str)
        + " | "
        + matched["gt_source"].astype(str)
    )
    counts = matched.groupby("gt_actor_key").size().sort_values(ascending=False)
    actor_keys = counts.head(max(1, int(args.postprocess_max_actors))).index.tolist()
    top = matched[matched["gt_actor_key"].isin(actor_keys)].copy()

    summary = (
        matched.groupby(["gt_actor_id", "gt_source", "gt_actor_type_id", "label"], dropna=False)
        .agg(
            samples=("frame_id", "count"),
            first_elapsed_s=("elapsed_s", "min"),
            last_elapsed_s=("elapsed_s", "max"),
            pred_mask_area_mean_px=("pred_mask_area_px", "mean"),
            pred_mask_area_std_px=("pred_mask_area_px", "std"),
            pred_bbox_area_mean_px=("pred_bbox_area_px", "mean"),
            pred_bbox_area_std_px=("pred_bbox_area_px", "std"),
            gt_bbox_area_mean_px=("gt_bbox_area_px", "mean"),
            gt_bbox_area_std_px=("gt_bbox_area_px", "std"),
            mean_iou=("match_iou", "mean"),
            mean_mask_to_gt_area_ratio=("pred_mask_to_gt_area_ratio", "mean"),
            std_mask_to_gt_area_ratio=("pred_mask_to_gt_area_ratio", "std"),
        )
        .reset_index()
    )
    summary["pred_mask_area_cv"] = (
        summary["pred_mask_area_std_px"] / summary["pred_mask_area_mean_px"].replace(0, np.nan)
    )
    summary["gt_bbox_area_cv"] = (
        summary["gt_bbox_area_std_px"] / summary["gt_bbox_area_mean_px"].replace(0, np.nan)
    )
    summary = summary.sort_values(["samples", "mean_iou"], ascending=[False, False])
    summary_path = output_dir / "object_size_variation_summary.csv"
    summary.to_csv(summary_path, index=False)

    figure = Figure(figsize=(12, 7), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots(1, 1)
    for actor_key in actor_keys:
        actor_df = top[top["gt_actor_key"] == actor_key].sort_values("elapsed_s")
        label = actor_key.split(" | ")[0]
        ax.plot(
            actor_df["elapsed_s"],
            actor_df["pred_mask_area_px"],
            linewidth=1.8,
            label=f"pred mask {label}",
        )
        ax.plot(
            actor_df["elapsed_s"],
            actor_df["gt_bbox_area_px"],
            linewidth=1.2,
            linestyle="--",
            label=f"GT bbox {label}",
        )
    ax.set_title("Segmentation Overlay Area vs CARLA Projected Ground Truth")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Area (px)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    _save_postprocess_figure(figure, output_dir / "overlay_area_vs_gt_over_time")

    figure = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots(1, 1)
    for actor_key in actor_keys:
        actor_df = top[top["gt_actor_key"] == actor_key].sort_values("elapsed_s")
        label = actor_key.split(" | ")[0]
        ax.plot(
            actor_df["elapsed_s"],
            actor_df["pred_mask_to_gt_area_ratio"],
            linewidth=1.8,
            label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="GT parity")
    ax.set_title("Segmentation Overlay Area / CARLA GT Area")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Area ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save_postprocess_figure(figure, output_dir / "overlay_to_gt_area_ratio_over_time")

    figure = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots(1, 1)
    for actor_key in actor_keys:
        actor_df = top[top["gt_actor_key"] == actor_key].sort_values("elapsed_s")
        label = actor_key.split(" | ")[0]
        ax.plot(actor_df["elapsed_s"], actor_df["match_iou"], linewidth=1.8, label=label)
    ax.set_title("2D Detection Box IoU vs CARLA Projected GT Box")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("2D IoU")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save_postprocess_figure(figure, output_dir / "bbox_iou_over_time")

    top_summary = summary.head(max(1, int(args.postprocess_max_actors))).copy()
    x = np.arange(len(top_summary))
    figure = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots(1, 1)
    ax.bar(x - 0.18, top_summary["pred_mask_area_cv"].fillna(0.0), width=0.36, label="Pred mask CV")
    ax.bar(x + 0.18, top_summary["gt_bbox_area_cv"].fillna(0.0), width=0.36, label="GT bbox CV")
    ax.set_title("Area Variation Coefficient by Matched CARLA Object")
    ax.set_xlabel("GT actor id")
    ax.set_ylabel("Coefficient of variation")
    ax.set_xticks(x)
    ax.set_xticklabels(top_summary["gt_actor_id"].astype(str), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    _save_postprocess_figure(figure, output_dir / "area_variation_summary")

    print(f"Wrote postprocess summary to {summary_path}")
    print(f"Wrote figures to {output_dir}")
    return 0


def _draw_overlay_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font_scale: float = 0.56,
    thickness: int = 2,
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
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _detection_count_summary(detections: Sequence[Dict[str, object]]) -> str:
    counts = Counter(str(item["label"]) for item in detections)
    parts = [f"{name} {counts[name]}" for name in ("vehicle", "person") if counts[name]]
    return ", ".join(parts) if parts else "none"


def draw_pole_segmentation_overlay(
    frame_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    detections: Sequence[Dict[str, object]],
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    *,
    args: argparse.Namespace,
    traffic_light_id: str,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    if mask is not None:
        if mask.shape[:2] != annotated.shape[:2]:
            mask = cv2.resize(
                mask,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        colors_rgb = trained_seg_demo.SEGMENTATION_OVERLAY_PALETTE_RGB[
            mask.clip(0, len(trained_seg_demo.SEGMENTATION_OVERLAY_PALETTE_RGB) - 1)
        ]
        colors_bgr = colors_rgb[:, :, ::-1]
        foreground = mask > 0
        strength = min(1.0, max(0.0, float(args.seg_mask_strength)))
        annotated[foreground] = (
            annotated[foreground].astype(np.float32) * (1.0 - strength)
            + colors_bgr[foreground].astype(np.float32) * strength
        ).astype(np.uint8)

        boundaries = trained_seg_demo._segmentation_mask_boundaries(mask)
        if boundaries.any():
            outline = cv2.dilate(
                boundaries.astype(np.uint8),
                trained_seg_demo.SEGMENTATION_BOUNDARY_KERNEL,
                iterations=1,
            ).astype(bool)
            outline &= foreground
            annotated[outline] = (0, 0, 0)
            annotated[boundaries] = (255, 255, 255)

    for detection in detections[:30]:
        x, y, width, height = detection["bbox"]
        color = tuple(int(channel) for channel in detection["color"])
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        label = f"{detection['label']} {int(detection['area'])}px"
        _draw_overlay_text(
            annotated,
            label,
            (x, max(18, y - 6)),
            font_scale=0.45,
            thickness=1,
        )

    payload_bytes = max(1, int(front_stats["payload_bytes"]))
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    compression_ratio = payload_bytes_uncompressed / payload_bytes
    lines = [
        f"Pole RGB split segmentation | traffic light {traffic_light_id}",
        f"Front half: {float(front_stats['front_ms']):.1f} ms",
        (
            "Feature payload: "
            f"{payload_bytes / 1024.0:.1f} KiB, "
            f"{payload_bytes_uncompressed / 1024.0:.1f} KiB baseline, "
            f"{compression_ratio:.2f}x"
        ),
        f"Detections: {len(detections)} ({_detection_count_summary(detections)})",
        mask_class_summary(mask, args) if mask is not None else "mask: waiting",
    ]
    if remote_stats is not None:
        lines.append(f"Back half: {float(remote_stats['server_ms']):.1f} ms")
        lines.append(f"Round trip: {float(remote_stats['round_trip_ms']):.1f} ms")
    else:
        lines.append("Back half: waiting")
        lines.append("Round trip: waiting")

    y = 28
    for line in lines:
        _draw_overlay_text(annotated, line, (10, y))
        y += 24
    return annotated


def build_plot_record(
    *,
    frame_id: int,
    elapsed_s: float,
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    detection_count: int,
) -> Dict[str, object]:
    return {
        "frame_id": int(frame_id),
        "elapsed_s": float(elapsed_s),
        "front_ms": float(front_stats["front_ms"]),
        "back_ms": float(remote_stats["server_ms"]) if remote_stats is not None else None,
        "round_trip_ms": (
            float(remote_stats["round_trip_ms"]) if remote_stats is not None else None
        ),
        "payload_bytes": int(front_stats["payload_bytes"]),
        "payload_bytes_uncompressed": int(front_stats["payload_bytes_uncompressed"]),
        "payload_kib": int(front_stats["payload_bytes"]) / 1024.0,
        "payload_uncompressed_kib": int(front_stats["payload_bytes_uncompressed"]) / 1024.0,
        "detections": int(detection_count),
    }


def write_pole_metrics_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    *,
    csv_path: Path,
    camera_width: int,
    camera_height: int,
    camera_resolution_label: str,
    seg_input_size: Tuple[int, int],
    weather_applied: Optional[str],
    town_loaded: str,
    traffic_light: "carla.Actor",
    camera_transform: "carla.Transform",
    detection_log_path: Optional[Path],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    od_id = _traffic_light_opendrive_id(traffic_light)
    manifest = {
        "wall_time_iso": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "run_tag": str(args.run_tag or ""),
        "csv_path": str(csv_path),
        "detection_log_path": str(detection_log_path or ""),
        "town_requested": str(args.town or ""),
        "town_loaded": town_loaded,
        "weather_preset_requested": str(args.weather_preset),
        "weather_preset_applied": weather_applied or WEATHER_PRESET_NONE,
        "traffic_light_id_requested": str(args.traffic_light_id),
        "traffic_light_actor_id": int(traffic_light.id),
        "traffic_light_opendrive_id": od_id or "",
        "camera_location_mode": str(args.camera_location_mode),
        "camera_transform": {
            "x": float(camera_transform.location.x),
            "y": float(camera_transform.location.y),
            "z": float(camera_transform.location.z),
            "pitch": float(camera_transform.rotation.pitch),
            "yaw": float(camera_transform.rotation.yaw),
            "roll": float(camera_transform.rotation.roll),
        },
        "camera_resolution_label": camera_resolution_label,
        "camera_width": int(camera_width),
        "camera_height": int(camera_height),
        "camera_fov": float(args.camera_fov),
        "fps": float(args.fps),
        "npc_vehicles": int(args.npc_vehicles),
        "npc_pedestrians": int(args.npc_pedestrians),
        "segmentation_model": str(args.segmentation_model),
        "seg_pretrained": bool(args.seg_pretrained),
        "seg_weights_path": str(args.seg_weights_path or ""),
        "trained_experiment_dir": str(args.trained_experiment_dir or ""),
        "seg_num_classes": int(args.seg_num_classes),
        "seg_class_scheme": str(args.seg_class_scheme),
        "seg_input_width": int(seg_input_size[0]),
        "seg_input_height": int(seg_input_size[1]),
        "metrics_warmup_frames": int(args.metrics_warmup_frames),
        "max_frames": int(args.max_frames),
        "run_duration_s": float(args.run_duration_s),
        "quantization_mode": str(args.quantization_mode),
        "entropy_coder": str(args.entropy_coder),
        "zstd_level": int(args.zstd_level),
        "roi_objectness_threshold": float(args.roi_objectness_threshold),
        "ae_mode": str(args.ae_mode),
        "ae_bottleneck_channels": int(args.ae_bottleneck_channels),
        "ae_spatial_stride": int(args.ae_spatial_stride),
        "ae_checkpoint": str(args.ae_checkpoint or ""),
        "ae_seed": int(args.ae_seed),
        "enable_semantic_gt": bool(args.enable_semantic_gt),
        "per_level_compress_probe": bool(args.per_level_compress_probe),
        "extra": trained_seg_demo._parse_manifest_extra(args.manifest_extra_json),
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _camera_blueprint(
    world: "carla.World",
    width: int,
    height: int,
    fov: float,
    fps: float,
) -> "carla.ActorBlueprint":
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(int(width)))
    camera_bp.set_attribute("image_size_y", str(int(height)))
    camera_bp.set_attribute("fov", str(float(fov)))
    camera_bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(fps))))
    return camera_bp


def _semantic_camera_blueprint(
    world: "carla.World",
    width: int,
    height: int,
    fov: float,
    fps: float,
) -> "carla.ActorBlueprint":
    gt_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    gt_bp.set_attribute("image_size_x", str(int(width)))
    gt_bp.set_attribute("image_size_y", str(int(height)))
    gt_bp.set_attribute("fov", str(float(fov)))
    gt_bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(fps))))
    return gt_bp


def _destroy_actors(actors: Iterable["carla.Actor"]) -> None:
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


def _close_split_runtime(
    *,
    stop_event: threading.Event,
    sockets: Sequence["od_collect.UDPMessageSocket"],
    remote_worker: seg_demo.SegmentationRemoteInferenceWorker,
    result_receiver: seg_demo.CameraResultReceiver,
    plot_sender: AsyncLivePlotSender,
) -> None:
    stop_event.set()
    remote_worker.join(timeout=1.0)
    result_receiver.join(timeout=1.0)
    for sock in sockets:
        try:
            sock.close()
        except Exception:
            pass
    plot_sender.close()


def run_client(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    if bool(args.list_traffic_lights):
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.load_world(args.town) if str(args.town).strip() else client.get_world()
        list_traffic_lights(world)
        return

    trained_seg_demo.apply_trained_checkpoint_args(args)
    front_device = od_demo.resolve_device(args.front_device)
    back_device = od_demo.resolve_device(args.back_device)
    camera_width, camera_height, camera_resolution_label = od_demo.resolve_camera_dimensions(args)
    seg_input_size = trained_seg_demo._resolve_seg_input_size(args, camera_width, camera_height)
    gui_enabled = od_demo.has_graphical_display() and not bool(args.headless)

    if front_device.type == "cuda" or back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    front_raw_model, back_raw_model = trained_seg_demo.build_segmentation_models(args)
    front_model = trained_seg_demo.TorchvisionSegmentationSplitModel(
        front_raw_model,
        front_device,
        input_size=seg_input_size,
    )
    back_model = trained_seg_demo.TorchvisionSegmentationSplitModel(
        back_raw_model,
        back_device,
        input_size=seg_input_size,
    )

    transport_cfg = od_collect.TransportConfig(
        quantization_mode=str(args.quantization_mode),
        entropy_coder_name=str(args.entropy_coder),
        zstd_level=int(args.zstd_level),
        roi_objectness_threshold=float(args.roi_objectness_threshold),
        bypass_rcnn_transform=False,
    )
    front_autoencoder = trained_seg_demo.build_per_level_autoencoder(args, front_device)
    back_autoencoder = trained_seg_demo.build_per_level_autoencoder(args, back_device)

    camera_sender = od_collect.UDPMessageSocket(
        bind_port=args.camera_source_port,
        remote_port=args.remote_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )
    remote_receiver = od_collect.UDPMessageSocket(
        bind_port=args.remote_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )
    remote_sender = od_collect.UDPMessageSocket(
        bind_port=args.remote_source_port,
        remote_port=args.camera_result_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )
    camera_receiver = od_collect.UDPMessageSocket(
        bind_port=args.camera_result_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        entropy_coder=transport_cfg.make_entropy_coder(),
    )

    stop_event = threading.Event()
    result_store = trained_seg_demo.SegmentationResultStore()
    split_camera = trained_seg_demo.CameraSideSegmentationSplitInference(
        front_model,
        camera_sender,
        transport=transport_cfg,
        autoencoder=front_autoencoder,
        per_level_compress_probe=bool(args.per_level_compress_probe),
    )
    remote_worker = trained_seg_demo.SegmentationRemoteInferenceWorker(
        model=back_model,
        receiver=remote_receiver,
        sender=remote_sender,
        device=back_device,
        stop_event=stop_event,
        transport=transport_cfg,
        autoencoder=back_autoencoder,
    )
    result_receiver = trained_seg_demo.CameraResultReceiver(
        receiver=camera_receiver,
        result_store=result_store,
        stop_event=stop_event,
    )
    remote_worker.start()
    result_receiver.start()

    plot_sender = AsyncLivePlotSender(args, gui_enabled=gui_enabled)
    if plot_sender.enabled():
        plot_sender.start()

    metrics_csv_path: Optional[Path] = None
    metrics_manifest_path: Optional[Path] = None
    metrics_collector: Optional[trained_seg_demo.AsyncMetricsCollector] = None
    if bool(args.collect_metrics):
        metrics_csv_path, metrics_manifest_path = trained_seg_demo.resolve_metrics_output_paths(args)
        metrics_collector = trained_seg_demo.AsyncMetricsCollector(
            csv_path=metrics_csv_path,
            enable_live_plot=False,
            gui_enabled=False,
            args=args,
        )
        metrics_collector.start()

    split_sockets = (camera_sender, remote_receiver, remote_sender, camera_receiver)
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.load_world(args.town) if str(args.town).strip() else client.get_world()
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

        traffic_light = resolve_traffic_light(world, str(args.traffic_light_id))
        camera_transform = build_camera_transform(traffic_light, args)
        anchor_location = traffic_light.get_transform().location
        weather_applied = trained_seg_demo.apply_weather_preset(world, str(args.weather_preset))

        original_settings = world.get_settings()
    except Exception:
        _close_split_runtime(
            stop_event=stop_event,
            sockets=split_sockets,
            remote_worker=remote_worker,
            result_receiver=result_receiver,
            plot_sender=plot_sender,
        )
        if metrics_collector is not None:
            metrics_collector.close()
        raise

    actors: List["carla.Actor"] = []
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
    gt_queue: Optional["queue.Queue[carla.Image]"] = None
    detection_logger: Optional[AsyncDetectionLogWriter] = None
    detection_log_path: Optional[Path] = None
    if bool(args.detection_log):
        detection_log_path = resolve_detection_log_path(args)
        detection_logger = AsyncDetectionLogWriter(detection_log_path, args)
        detection_logger.start()

    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"World: {world.get_map().name}")
    print(f"Weather: {weather_applied or WEATHER_PRESET_NONE}")
    print(f"Traffic light actor id: {traffic_light.id}")
    od_id = _traffic_light_opendrive_id(traffic_light)
    if od_id:
        print(f"Traffic light OpenDRIVE id: {od_id}")
    print(
        "Pole camera transform: "
        f"loc=({camera_transform.location.x:.2f}, {camera_transform.location.y:.2f}, "
        f"{camera_transform.location.z:.2f}), "
        f"pitch={camera_transform.rotation.pitch:.1f}, "
        f"yaw={camera_transform.rotation.yaw:.1f}, "
        f"roll={camera_transform.rotation.roll:.1f}"
    )
    print(f"Camera resolution: {camera_width}x{camera_height} ({camera_resolution_label})")
    print(f"Segmentation model: {args.segmentation_model}, input={seg_input_size[0]}x{seg_input_size[1]}")
    print(f"Segmentation classes: {args.seg_class_scheme}, num_classes={args.seg_num_classes}")
    print(f"Segmentation checkpoint: {args.seg_weights_path or '(none)'}")
    print(f"Front device: {front_device}, back device: {back_device}")
    print(
        "UDP ports: "
        f"camera {args.camera_source_port} -> remote {args.remote_port}, "
        f"remote {args.remote_source_port} -> camera {args.camera_result_port}"
    )
    if plot_sender.warning:
        print(plot_sender.warning)
    if detection_log_path is not None:
        print(f"Object-size detection log: {detection_log_path}")
    if metrics_csv_path is not None:
        print(f"Metrics CSV: {metrics_csv_path}")
    else:
        print("Metrics data collection disabled. CSV logging is off.")
    if metrics_collector is not None and metrics_collector.warning:
        print(metrics_collector.warning)
    if not gui_enabled:
        if args.headless:
            print("GUI disabled by --headless. Running without OpenCV or matplotlib windows.")
        else:
            print("No graphical display detected. Running without OpenCV or matplotlib windows.")

    try:
        if bool(args.sync_world):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / max(0.1, float(args.fps))
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.tick()
        else:
            traffic_manager.set_synchronous_mode(False)

        background_vehicles = spawn_background_vehicles_near(
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

        pedestrians, pedestrian_controllers = spawn_background_pedestrians_near(
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
            _camera_blueprint(world, camera_width, camera_height, args.camera_fov, args.fps),
            camera_transform,
        )
        actors.append(camera)
        camera.listen(lambda image: od_demo.put_latest(image_queue, image))

        if bool(args.enable_semantic_gt):
            gt_camera = world.spawn_actor(
                _semantic_camera_blueprint(world, camera_width, camera_height, args.camera_fov, args.fps),
                camera_transform,
            )
            actors.append(gt_camera)
            gt_queue = queue.Queue(maxsize=2)
            gt_camera.listen(lambda image, q=gt_queue: od_demo.put_latest(q, image))
            print("Semantic GT camera enabled (3-class mIoU logging is on).")

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

        static_gt_objects: List[Dict[str, object]] = []
        if detection_logger is not None and bool(args.include_static_level_bboxes):
            static_gt_objects = project_ground_truth_objects(
                world,
                camera,
                width=camera_width,
                height=camera_height,
                fov=float(args.camera_fov),
                max_distance_m=float(args.gt_max_distance),
                include_static_level_bboxes=True,
                include_dynamic_actors=False,
            )
            if static_gt_objects:
                print(f"Projected {len(static_gt_objects)} static CARLA level GT boxes.")

        if gui_enabled:
            cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        else:
            print("Headless run active. Press Ctrl+C to stop.")

        start_perf = time.perf_counter()
        processed_frames = 0
        metrics_start_perf: Optional[float] = None
        measurement_frames_logged = 0
        max_measurement_frames = max(0, int(args.max_frames))
        run_duration_s = max(0.0, float(args.run_duration_s))
        needs_measurement_window = (
            metrics_collector is not None or max_measurement_frames > 0 or run_duration_s > 0.0
        )
        metrics_warmup_remaining = (
            max(0, int(args.metrics_warmup_frames)) if needs_measurement_window else 0
        )
        if metrics_warmup_remaining == 0:
            metrics_start_perf = time.perf_counter()
        elif metrics_collector is not None:
            print(
                "Metrics warm-up: skipping the first "
                f"{metrics_warmup_remaining} frame(s) while feature range trackers stabilize."
            )

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
                print(
                    f"Warning: camera frame was not received within {args.camera_timeout:.1f}s; retrying."
                )
                continue
            world_frame = int(world_frame) if bool(args.sync_world) else int(image.frame)

            frame_bgr = od_demo.camera_image_to_bgr(image)
            front_stats = split_camera.process(int(image.frame), frame_bgr)

            gt_3class: Optional[np.ndarray] = None
            if gt_queue is not None:
                if bool(args.sync_world):
                    gt_image = od_demo.wait_for_camera_frame(
                        gt_queue,
                        world_frame,
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
                else:
                    print(
                        "Warning: semantic-GT frame was not received; "
                        "mIoU columns will be NaN for this frame."
                    )

            result = result_store.wait_for(
                int(image.frame),
                float(args.result_timeout),
                tick_callback=None,
                tick_hz=max(0.1, float(args.fps)),
            )

            remote_stats = None
            mask: Optional[np.ndarray] = None
            if result is not None:
                remote_stats = {
                    "server_ms": float(result["server_ms"]),
                    "round_trip_ms": (
                        time.perf_counter() - float(result["camera_sent_perf"])
                    )
                    * 1000.0,
                }
                mask = result.get("mask") if isinstance(result.get("mask"), np.ndarray) else None

            detections = summarize_segmentation_detections(mask, int(args.min_detection_area), args)
            elapsed_s = time.perf_counter() - start_perf
            if detection_logger is not None:
                gt_objects = project_ground_truth_objects(
                    world,
                    camera,
                    width=camera_width,
                    height=camera_height,
                    fov=float(args.camera_fov),
                    max_distance_m=float(args.gt_max_distance),
                    include_static_level_bboxes=False,
                    include_dynamic_actors=True,
                )
                if static_gt_objects:
                    gt_objects.extend(static_gt_objects)
                detection_logger.submit_many(
                    build_detection_log_rows(
                        frame_id=int(image.frame),
                        elapsed_s=elapsed_s,
                        detections=detections,
                        gt_objects=gt_objects,
                        min_iou=float(args.gt_match_iou_threshold),
                    )
                )
            processed_frames += 1
            if needs_measurement_window:
                if metrics_warmup_remaining > 0:
                    metrics_warmup_remaining -= 1
                    if metrics_warmup_remaining == 0:
                        metrics_start_perf = time.perf_counter()
                else:
                    if metrics_start_perf is None:
                        metrics_start_perf = time.perf_counter()
                    metrics_elapsed_s = time.perf_counter() - metrics_start_perf
                    if metrics_collector is not None:
                        metrics_collector.submit(
                            trained_seg_demo.build_metrics_record(
                                frame_id=int(image.frame),
                                elapsed_s=metrics_elapsed_s,
                                args=args,
                                front_stats=front_stats,
                                remote_stats=remote_stats,
                                mask=mask,
                                camera_width=camera_width,
                                camera_height=camera_height,
                                camera_resolution_label=camera_resolution_label,
                                seg_input_size=seg_input_size,
                                town=world.get_map().name,
                                weather_preset=weather_applied or WEATHER_PRESET_NONE,
                                gt_3class=gt_3class,
                            )
                        )
                    measurement_frames_logged += 1
                    if (
                        max_measurement_frames > 0
                        and measurement_frames_logged >= max_measurement_frames
                    ):
                        print(f"Reached --max-frames={max_measurement_frames}; stopping run.")
                        break
                    if run_duration_s > 0.0 and metrics_elapsed_s >= run_duration_s:
                        print(f"Reached --run-duration-s={run_duration_s:.1f}; stopping run.")
                        break
            if (
                plot_sender.enabled()
                and processed_frames % max(1, int(args.live_plot_update_interval)) == 0
            ):
                plot_sender.submit(
                    build_plot_record(
                        frame_id=int(image.frame),
                        elapsed_s=elapsed_s,
                        front_stats=front_stats,
                        remote_stats=remote_stats,
                        detection_count=len(detections),
                    )
                )

            if gui_enabled:
                annotated = draw_pole_segmentation_overlay(
                    frame_bgr,
                    mask,
                    detections,
                    front_stats,
                    remote_stats,
                    args=args,
                    traffic_light_id=str(args.traffic_light_id),
                )
                cv2.imshow(DEFAULT_WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        stop_event.set()
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        if bool(args.sync_world):
            try:
                world.apply_settings(original_settings)
            except RuntimeError:
                pass

        _destroy_actors(actors)
        _close_split_runtime(
            stop_event=stop_event,
            sockets=split_sockets,
            remote_worker=remote_worker,
            result_receiver=result_receiver,
            plot_sender=plot_sender,
        )
        if detection_logger is not None:
            detection_logger.close()
        if detection_log_path is not None:
            print(f"Saved object-size detection log to {detection_log_path}")
        if metrics_collector is not None:
            metrics_collector.close()
        if metrics_csv_path is not None:
            print(f"Saved metrics CSV to {metrics_csv_path}")
        if metrics_manifest_path is not None and metrics_csv_path is not None:
            try:
                write_pole_metrics_manifest(
                    metrics_manifest_path,
                    args,
                    csv_path=metrics_csv_path,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    camera_resolution_label=camera_resolution_label,
                    seg_input_size=seg_input_size,
                    weather_applied=weather_applied,
                    town_loaded=world.get_map().name,
                    traffic_light=traffic_light,
                    camera_transform=camera_transform,
                    detection_log_path=detection_log_path,
                )
                print(f"Saved metrics manifest to {metrics_manifest_path}")
            except Exception as exc:
                print(f"Warning: unable to write metrics manifest: {exc}", file=sys.stderr)
        if gui_enabled:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if str(args.postprocess_detection_log or "").strip():
        raise SystemExit(run_detection_log_postprocess(args))
    if args.metrics_plot_worker:
        raise SystemExit(run_metrics_plot_worker(args))
    run_client(args)


if __name__ == "__main__":
    main()

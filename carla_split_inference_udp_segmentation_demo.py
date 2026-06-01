#!/usr/bin/env python3

"""
CARLA split-inference UDP demo for real-time semantic segmentation.

This is a segmentation-oriented sibling of carla_split_inference_udp_demo.py.
It keeps the localhost UDP data flow and feature compression path, but swaps
the Faster R-CNN detector for torchvision's LR-ASPP MobileNetV3 segmentation
network. The camera side runs the backbone and sends compressed feature maps;
the remote side runs the segmentation classifier head and sends the predicted
mask back for overlay on the rendered camera image.

No Dedelayed feature prediction/fusion path is included in this script.
Press `q` or `Esc` in the OpenCV view to exit.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import queue
import random
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import carla_split_inference_udp_demo as od_demo
import carla_split_inference_udp_data_collect as od_collect

carla = od_demo.carla
cv2 = od_demo.cv2

SEGMENTATION_MODEL_LRASPP = "lraspp_mobilenet_v3_large"
SEGMENTATION_MODEL_CHOICES = (SEGMENTATION_MODEL_LRASPP,)

VOC_SEGMENTATION_LABELS = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

SEGMENTATION_OVERLAY_PALETTE_RGB = np.array(
    [
        (0, 0, 0),
        (255, 64, 64),
        (0, 255, 128),
        (255, 224, 0),
        (0, 160, 255),
        (255, 64, 255),
        (255, 144, 0),
        (0, 240, 255),
        (176, 96, 255),
        (128, 255, 0),
        (255, 0, 128),
        (0, 255, 255),
        (255, 192, 64),
        (64, 255, 64),
        (192, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (255, 96, 0),
        (96, 255, 192),
        (255, 32, 32),
        (160, 160, 255),
    ],
    dtype=np.uint8,
)
SEGMENTATION_BOUNDARY_KERNEL = np.ones((3, 3), dtype=np.uint8)

DEFAULT_METRICS_LOG_DIR = Path(__file__).resolve().parent / "metrics_logs"
DEFAULT_LIVE_PLOT_REFRESH_SECONDS = od_demo.DEFAULT_LIVE_PLOT_REFRESH_SECONDS
WEATHER_PRESET_NONE = "unchanged"
METRICS_CSV_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "frame_id",
    "run_tag",
    "town",
    "weather_preset",
    "camera_resolution_label",
    "camera_width",
    "camera_height",
    "camera_fov",
    "npc_vehicles",
    "npc_pedestrians",
    "segmentation_model",
    "seg_input_width",
    "seg_input_height",
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "payload_bytes",
    "payload_bytes_uncompressed",
    "payload_kib",
    "payload_uncompressed_kib",
    "payload_chunks",
    "mask_available",
    "mask_foreground_pixels",
    "mask_classes",
    # Transport-axis tags: mirror the columns in carla_split_inference_udp_data_collect
    # so the OD and segmentation CSVs can be concatenated for cross-pipeline analysis.
    # roi_drop_fraction_total is the actual fraction of FPN cells zeroed by the
    # saliency gate (computed per frame); per-level fractions go in the JSON column.
    "quantization_mode",
    "entropy_coder",
    "zstd_level",
    "roi_objectness_threshold",
    "roi_drop_fraction_total",
    "roi_drop_fraction_per_level_json",
    "ae_mode",
    "ae_bottleneck_channels",
    "ae_spatial_stride",
    # Quality vs ground truth (populated when --enable-semantic-gt is set).
    # The 3-class scheme is {0=background, 1=vehicle, 2=person} and is
    # documented alongside the mapping helpers below.
    "gt_camera_available",
    "miou_binary",
    "miou_3class_macro",
    "miou_vehicle_iou",
    "miou_person_iou",
    "gt_vehicle_pixels",
    "gt_person_pixels",
    # Per-level byte breakdown (populated when --per-level-compress-probe is set).
    # JSON-encoded {level_name: bytes} so the schema stays stable when the
    # backbone level set changes.
    "per_level_uncompressed_bytes_json",
    "per_level_compressed_bytes_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CARLA split semantic segmentation over localhost UDP."
    )
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
        "--drive-mode",
        choices=("autopilot", "manual"),
        default="autopilot",
        help="Control mode for the ego vehicle.",
    )
    parser.add_argument(
        "--manual-throttle",
        type=float,
        default=0.1,
        help="Per-frame throttle increment while W or Up is held in manual mode.",
    )
    parser.add_argument(
        "--manual-brake",
        type=float,
        default=0.2,
        help="Per-frame brake increment while S or Down is held in manual mode.",
    )
    parser.add_argument(
        "--manual-steer-step",
        type=float,
        default=5e-4,
        help="Steering increment per elapsed millisecond while steering keys are held.",
    )
    parser.add_argument(
        "--camera-resolution",
        choices=["custom", *od_demo.CAMERA_RESOLUTION_PRESETS.keys()],
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
        "--segmentation-model",
        choices=SEGMENTATION_MODEL_CHOICES,
        default=SEGMENTATION_MODEL_LRASPP,
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
    parser.set_defaults(seg_pretrained=True, collect_metrics=True, live_plot=True)
    parser.add_argument(
        "--seg-weights-path",
        default="",
        help="Optional state_dict checkpoint for the segmentation model.",
    )
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
        default=0.72,
        help="Visualization overlay strength for predicted segmentation masks.",
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
    parser.add_argument(
        "--metrics-log-dir",
        default=str(DEFAULT_METRICS_LOG_DIR),
        help="Directory where the segmentation metrics CSV will be saved.",
    )
    parser.add_argument(
        "--metrics-log-prefix",
        default="split_segmentation_metrics",
        help="Filename prefix used for the metrics CSV.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional short tag stamped into every CSV row and the manifest.",
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
        help="Optional JSON object merged into the run manifest for sweep metadata.",
    )
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
        help="Enable metrics CSV logging.",
    )
    metrics_collection_group.add_argument(
        "--disable-data-collection",
        "--disable-metrics-collection",
        dest="collect_metrics",
        action="store_false",
        help="Disable metrics CSV logging.",
    )
    parser.add_argument(
        "--metrics-warmup-frames",
        type=int,
        default=od_demo.DEFAULT_METRICS_WARMUP_FRAMES,
        help="Number of initial frames excluded while feature range trackers stabilize.",
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
        default=od_demo.DEFAULT_METRICS_BATCH_SIZE,
        help="Number of queued metrics samples written to CSV per batch flush.",
    )
    parser.add_argument(
        "--metrics-flush-interval",
        type=float,
        default=od_demo.DEFAULT_METRICS_FLUSH_INTERVAL,
        help="Maximum seconds between background CSV flushes.",
    )
    parser.add_argument(
        "--metrics-queue-size",
        type=int,
        default=2048,
        help="Maximum number of queued metrics samples before old samples are dropped.",
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
        "--headless",
        action="store_true",
        help="Disable the OpenCV camera view.",
    )
    # ------------------------------------------------------------------
    # Transport-layer sweep knobs. These mirror the four axes already in
    # carla_split_inference_udp_data_collect (quantization, entropy coder,
    # ROI/saliency gate, autoencoder bottleneck) so the segmentation pipeline
    # can be characterised against the same compression-headroom dimensions
    # as the object-detection pipeline. The ROI gate has no RPN to lean on,
    # so it acts on per-cell L2-saliency of the backbone features and the
    # threshold is interpreted as a target *drop fraction* (q in [0,1)) of
    # the lowest-saliency cells, which is directly comparable to the
    # roi_drop_fraction_total reported by the OD sweep.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--quantization-mode",
        choices=od_collect.QUANT_MODE_CHOICES,
        default=od_collect.QUANT_MODE_PER_TENSOR_UINT8,
        help=(
            "How feature tensors are quantized before serialization. "
            "Same options as the OD sweep so payloads are directly comparable."
        ),
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
        help="zstd compression level when --entropy-coder=zstd (1..22).",
    )
    parser.add_argument(
        "--roi-objectness-threshold",
        type=float,
        default=0.0,
        help=(
            "Saliency-gate target drop fraction in [0,1). The front side "
            "computes per-cell L2 norms across channels and zeros the "
            "lowest-saliency q fraction of cells (per FPN level). 0 disables "
            "the gate. The achieved fraction is logged as "
            "roi_drop_fraction_total so the OD and segmentation rows can be "
            "aligned."
        ),
    )
    parser.add_argument(
        "--ae-mode",
        choices=od_collect.AE_MODE_CHOICES,
        default=od_collect.AE_MODE_OFF,
        help=(
            "Optional 1x1-conv autoencoder bottleneck applied per backbone "
            "level before the wire. random_projection uses a deterministic "
            "seeded init; checkpoint loads a trained state dict from "
            "--ae-checkpoint."
        ),
    )
    parser.add_argument(
        "--ae-bottleneck-channels",
        type=int,
        default=64,
        help="Channel count at the autoencoder bottleneck (per level).",
    )
    parser.add_argument(
        "--ae-spatial-stride",
        type=int,
        default=1,
        help=(
            "Spatial stride applied by the autoencoder encoder (matched by "
            "the decoder via output_size). 1 keeps spatial size; 2 halves it."
        ),
    )
    parser.add_argument(
        "--ae-checkpoint",
        default="",
        help=(
            "Path to a torch.save() blob containing per-level autoencoder "
            "state dicts keyed by FPN level name."
        ),
    )
    parser.add_argument(
        "--ae-seed",
        type=int,
        default=0,
        help="Seed used when --ae-mode=random_projection so both sides match.",
    )
    # ------------------------------------------------------------------
    # Quality + per-level instrumentation. Off by default so existing
    # sweep scripts keep producing the same data; the cross-pipeline
    # driver flips both on.
    # ------------------------------------------------------------------
    semantic_gt_group = parser.add_mutually_exclusive_group()
    semantic_gt_group.add_argument(
        "--enable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_true",
        help=(
            "Spawn a co-located CARLA semantic-segmentation camera and log "
            "per-frame mIoU between the LR-ASPP predicted mask and the "
            "CARLA-rendered ground truth. Mapped to a 3-class scheme "
            "{background, vehicle, person} to bridge VOC and Cityscapes."
        ),
    )
    semantic_gt_group.add_argument(
        "--disable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_false",
        help="Skip the GT-camera + mIoU instrumentation (default).",
    )
    parser.set_defaults(enable_semantic_gt=False, per_level_compress_probe=False)
    parser.add_argument(
        "--per-level-compress-probe",
        dest="per_level_compress_probe",
        action="store_true",
        help=(
            "Record per-level uncompressed and individually-compressed byte "
            "counts in the metrics CSV (per_level_*_bytes_json). Mirrors the "
            "OD data-collect demo so the cross-pipeline analyzer can "
            "decompose payload by backbone level."
        ),
    )
    return parser.parse_args()


def _resolve_seg_input_size(
    args: argparse.Namespace,
    camera_width: int,
    camera_height: int,
) -> Tuple[int, int]:
    width = int(args.seg_input_width) if int(args.seg_input_width) > 0 else int(camera_width)
    height = int(args.seg_input_height) if int(args.seg_input_height) > 0 else int(camera_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid segmentation input size {width}x{height}")
    return width, height


def apply_weather_preset(world: "carla.World", preset_name: str) -> Optional[str]:
    if not preset_name or preset_name == WEATHER_PRESET_NONE:
        return None
    preset = getattr(carla.WeatherParameters, preset_name, None)
    if preset is None:
        print(
            f"Warning: unknown weather preset {preset_name!r}; leaving CARLA weather unchanged.",
            file=sys.stderr,
        )
        return None
    try:
        world.set_weather(preset)
        return preset_name
    except RuntimeError as exc:
        print(f"Warning: failed to apply weather preset {preset_name}: {exc}", file=sys.stderr)
        return None


def _load_state_dict_if_requested(model: torch.nn.Module, path: str) -> None:
    if not path:
        return
    state = torch.load(Path(path).expanduser(), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Segmentation checkpoint {path!r} did not contain a state_dict.")
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Warning: segmentation checkpoint keys did not exactly match.")
        if incompatible.missing_keys:
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")


def _build_raw_lraspp_model(args: argparse.Namespace) -> torch.nn.Module:
    try:
        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )
    except Exception as exc:
        raise RuntimeError(
            "torchvision semantic segmentation models are required for this client."
        ) from exc

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if bool(args.seg_pretrained) else None
    try:
        if weights is None:
            model = lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=len(VOC_SEGMENTATION_LABELS),
            )
        else:
            model = lraspp_mobilenet_v3_large(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load LR-ASPP MobileNetV3. The first pretrained run may "
            "need internet access. Re-run with --seg-disable-pretrained or pass "
            "--seg-weights-path."
        ) from exc
    _load_state_dict_if_requested(model, str(args.seg_weights_path or ""))
    return model.eval()


def build_segmentation_models(args: argparse.Namespace) -> Tuple[torch.nn.Module, torch.nn.Module]:
    if args.segmentation_model != SEGMENTATION_MODEL_LRASPP:
        raise ValueError(f"Unsupported segmentation model {args.segmentation_model!r}")
    front_model = _build_raw_lraspp_model(args)
    back_model = _build_raw_lraspp_model(
        argparse.Namespace(**{**vars(args), "seg_pretrained": False, "seg_weights_path": ""})
    )
    back_model.load_state_dict(front_model.state_dict())
    back_model.eval()
    return front_model, back_model


class TorchvisionSegmentationSplitModel:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        input_size: Tuple[int, int],
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.input_width, self.input_height = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (self.input_width, self.input_height):
            rgb = cv2.resize(
                rgb,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
            / 255.0
        )
        return (tensor - self.mean) / self.std

    def encode(self, image_tensor: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        features = self.model.backbone(image_tensor)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        return OrderedDict((str(name), tensor) for name, tensor in features.items())

    def decode_logits(
        self,
        features: "OrderedDict[str, torch.Tensor]",
        *,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        logits = self.model.classifier(features)
        if isinstance(logits, dict):
            logits = logits["out"]
        if tuple(logits.shape[-2:]) != (self.input_height, self.input_width):
            logits = F.interpolate(
                logits,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
        if tuple(output_size) != (self.input_height, self.input_width):
            logits = F.interpolate(
                logits,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def decode_mask(
        self,
        features: "OrderedDict[str, torch.Tensor]",
        *,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        logits = self.decode_logits(features, output_size=output_size)
        return logits.argmax(dim=1).squeeze(0).detach().to("cpu").numpy().astype(np.uint8)


class PerLevelFeatureAutoencoder:
    """Holds one od_collect.FeatureAutoencoder per backbone level.

    LR-ASPP MobileNetV3 emits two backbone levels (`low`, `high`) with
    different channel counts (40 and 960), so a single shared encoder/decoder
    pair (as in the OD sweep, where every FPN level has 256 channels) does not
    apply. We lazily build a per-level pair the first time a level is seen,
    sized to that level's channel count, and reuse it for the rest of the run.
    """

    def __init__(
        self,
        *,
        mode: str,
        bottleneck_channels: int,
        spatial_stride: int,
        seed: int,
        checkpoint_path: str,
        device: torch.device,
    ) -> None:
        self.mode = str(mode)
        self.bottleneck_channels = int(bottleneck_channels)
        self.spatial_stride = int(spatial_stride)
        self.seed = int(seed)
        self.checkpoint_path = str(checkpoint_path or "")
        self.device = device
        self._levels: Dict[str, "od_collect.FeatureAutoencoder"] = {}
        self._checkpoint_state: Optional[Dict[str, object]] = None
        if self.mode == od_collect.AE_MODE_CHECKPOINT:
            if not self.checkpoint_path:
                raise ValueError(
                    "PerLevelFeatureAutoencoder: AE_MODE_CHECKPOINT requires --ae-checkpoint=PATH"
                )
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("state_dict", "model", "model_state_dict"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            if not isinstance(state, dict):
                raise ValueError(
                    f"AE checkpoint {self.checkpoint_path!r} did not contain a state_dict."
                )
            self._checkpoint_state = state

    def _build_for_level(self, name: str, in_channels: int) -> "od_collect.FeatureAutoencoder":
        bottleneck = max(1, min(self.bottleneck_channels, int(in_channels)))
        stride = max(1, self.spatial_stride)
        ae = od_collect.FeatureAutoencoder(
            in_channels=int(in_channels),
            bottleneck_channels=bottleneck,
            spatial_stride=stride,
        )
        if self.mode == od_collect.AE_MODE_RANDOM_PROJECTION:
            # Seed deterministically per level so the sender and receiver agree.
            level_seed = self.seed ^ (hash(name) & 0xFFFFFFFF)
            generator = torch.Generator(device="cpu").manual_seed(int(level_seed))
            with torch.no_grad():
                for parameter in ae.parameters():
                    if parameter.dim() >= 2:
                        torch.nn.init.normal_(parameter, mean=0.0, std=0.05, generator=generator)
                    else:
                        torch.nn.init.zeros_(parameter)
        elif self.mode == od_collect.AE_MODE_CHECKPOINT:
            assert self._checkpoint_state is not None
            sub = self._checkpoint_state.get(name)
            if not isinstance(sub, dict):
                raise ValueError(
                    f"AE checkpoint missing state for backbone level {name!r}; "
                    f"expected a dict keyed by level name."
                )
            missing, unexpected = ae.load_state_dict(sub, strict=False)
            if missing:
                print(f"AE level {name!r} checkpoint missing {len(missing)} keys", file=sys.stderr)
            if unexpected:
                print(f"AE level {name!r} checkpoint had {len(unexpected)} unexpected keys", file=sys.stderr)
        return ae.to(self.device).eval()

    def _level(self, name: str, in_channels: int) -> "od_collect.FeatureAutoencoder":
        ae = self._levels.get(name)
        if ae is None:
            ae = self._build_for_level(name, in_channels)
            self._levels[name] = ae
        return ae

    def encode(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        in_channels = int(tensor.shape[1])
        return self._level(name, in_channels).encode(tensor.to(self.device))

    def decode(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        original_channels: int,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return self._level(name, original_channels).decode(tensor.to(self.device), output_size=output_size)


def build_per_level_autoencoder(
    args: argparse.Namespace, device: torch.device
) -> Optional[PerLevelFeatureAutoencoder]:
    if args.ae_mode == od_collect.AE_MODE_OFF:
        return None
    return PerLevelFeatureAutoencoder(
        mode=args.ae_mode,
        bottleneck_channels=int(args.ae_bottleneck_channels),
        spatial_stride=int(args.ae_spatial_stride),
        seed=int(args.ae_seed),
        checkpoint_path=str(args.ae_checkpoint or ""),
        device=device,
    )


# ---------------------------------------------------------------------------
# Ground-truth semantic camera + mIoU.
#
# CARLA's `sensor.camera.semantic_segmentation` encodes the per-pixel class id
# in the R channel of the BGRA buffer. CARLA uses a Cityscapes-style scheme
# (Pedestrian=12, Rider=13, Car=14, Truck=15, Bus=16, Train=17, Motorcycle=18,
# Bicycle=19); the LR-ASPP backbone is pretrained on Pascal VOC's 21-class
# scheme, so a class-by-class mapping isn't meaningful. We compress both to a
# common 3-class scheme {0=background, 1=vehicle, 2=person} that captures
# the two object types this testbed cares about and lets us report a
# defensible macro-mIoU.
# ---------------------------------------------------------------------------

# CARLA Cityscapes-style tag ids -> 3-class id.
_CARLA_TAG_TO_3CLASS: Dict[int, int] = {
    12: 2,  # Pedestrian
    13: 2,  # Rider (treated as person)
    14: 1,  # Car
    15: 1,  # Truck
    16: 1,  # Bus
    17: 1,  # Train
    18: 1,  # Motorcycle
    19: 1,  # Bicycle
}

# Pascal VOC class ids (matches VOC_SEGMENTATION_LABELS) -> 3-class id.
_VOC_LABEL_TO_3CLASS: Dict[int, int] = {
    1: 1,   # aeroplane (treated as vehicle)
    2: 1,   # bicycle
    4: 1,   # boat
    6: 1,   # bus
    7: 1,   # car
    14: 1,  # motorbike
    19: 1,  # train
    15: 2,  # person
}

CLASS_ID_BACKGROUND = 0
CLASS_ID_VEHICLE = 1
CLASS_ID_PERSON = 2
CLASS_NAMES_3CLASS = ("background", "vehicle", "person")


def _build_lookup(table: Dict[int, int], size: int = 256) -> np.ndarray:
    lut = np.zeros(size, dtype=np.uint8)
    for raw, mapped in table.items():
        if 0 <= raw < size:
            lut[raw] = mapped
    return lut


_CARLA_LUT = _build_lookup(_CARLA_TAG_TO_3CLASS)
_VOC_LUT = _build_lookup(_VOC_LABEL_TO_3CLASS)


def carla_semantic_image_to_tags(image: "carla.Image") -> np.ndarray:
    """Decode a CARLA semantic-segmentation image into a (H, W) uint8 tag map.

    Mirrors the testbed convention used elsewhere (R channel of BGRA carries
    the raw Cityscapes-style tag id).
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)
    )
    return arr[:, :, 2].copy()


def map_carla_tags_to_3class(tags: np.ndarray) -> np.ndarray:
    return _CARLA_LUT[tags]


def map_voc_labels_to_3class(mask: np.ndarray) -> np.ndarray:
    return _VOC_LUT[mask]


def compute_3class_iou(
    pred_3class: np.ndarray,
    gt_3class: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return (vehicle_iou, person_iou, macro_iou, binary_iou).

    Macro-mIoU is averaged over classes whose union is non-zero, including
    background. This is the more honest reduction when scenes are mostly
    background; classes that aren't present in either prediction or GT for
    a given frame are dropped from the average rather than skewing it down
    to zero. Returns NaN for any class with empty union (and for the
    macro/binary numbers if every class is empty).
    """
    if pred_3class.shape != gt_3class.shape:
        # Predicted mask comes back at the camera's output_size; gt is at the
        # GT camera's resolution. They are spawned at the same size, but
        # rounding via the LR-ASPP back-half can leave them off by a row or
        # column in degenerate cases. Resize the gt to the predicted shape.
        gt_3class = cv2.resize(
            gt_3class,
            (pred_3class.shape[1], pred_3class.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    ious: List[float] = []
    per_class: Dict[int, float] = {}
    for class_id in (CLASS_ID_BACKGROUND, CLASS_ID_VEHICLE, CLASS_ID_PERSON):
        pred_match = pred_3class == class_id
        gt_match = gt_3class == class_id
        union = int(np.logical_or(pred_match, gt_match).sum())
        if union == 0:
            per_class[class_id] = float("nan")
            continue
        intersection = int(np.logical_and(pred_match, gt_match).sum())
        iou = intersection / union
        per_class[class_id] = iou
        ious.append(iou)

    macro_iou = float(np.mean(ious)) if ious else float("nan")

    pred_fg = pred_3class != CLASS_ID_BACKGROUND
    gt_fg = gt_3class != CLASS_ID_BACKGROUND
    union_fg = int(np.logical_or(pred_fg, gt_fg).sum())
    binary_iou = (
        int(np.logical_and(pred_fg, gt_fg).sum()) / union_fg
        if union_fg > 0
        else float("nan")
    )

    return (
        per_class[CLASS_ID_VEHICLE],
        per_class[CLASS_ID_PERSON],
        macro_iou,
        binary_iou,
    )


def saliency_drop_masks(
    features: "OrderedDict[str, torch.Tensor]",
    drop_fraction: float,
) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, float]]:
    """Compute a per-level binary keep-mask by dropping the lowest-q cells by L2 norm.

    Mirrors the OD pipeline's RPN-objectness gate. For LR-ASPP we have no
    RPN, so we approximate "this FPN cell is unlikely to matter" with the
    per-cell L2 norm of the backbone features (smaller norm -> less salient).
    The user-supplied threshold is treated as the target drop fraction q so
    the achieved drop fraction matches by construction (modulo ties), which
    keeps it directly aligned with the OD sweep's roi_drop_fraction_total.
    Returns (masks, total_drop_fraction, per_level_drop_fraction).
    """
    if drop_fraction <= 0.0 or not features:
        return {}, 0.0, {}
    q = float(min(max(drop_fraction, 0.0), 0.999))
    masks: Dict[str, torch.Tensor] = {}
    kept_total = 0
    cells_total = 0
    per_level: Dict[str, float] = {}
    for name, tensor in features.items():
        if tensor.ndim != 4:
            continue
        # tensor: (B, C, H, W) -> per-cell L2 across channels.
        saliency = torch.linalg.vector_norm(tensor, dim=1)  # (B, H, W)
        flat = saliency.reshape(saliency.shape[0], -1)
        # quantile across spatial cells, per batch element.
        thresholds = torch.quantile(flat, q, dim=1, keepdim=True)
        keep = flat > thresholds
        kept_per_level = int(keep.sum().item())
        cells_per_level = int(keep.numel())
        masks[name] = keep.reshape_as(saliency)
        kept_total += kept_per_level
        cells_total += cells_per_level
        per_level[name] = (
            1.0 - kept_per_level / cells_per_level if cells_per_level > 0 else 0.0
        )
    drop_total = 1.0 - (kept_total / cells_total) if cells_total > 0 else 0.0
    return masks, drop_total, per_level


class SegmentationResultStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._results: Dict[int, Dict[str, object]] = {}

    def put(self, frame_id: int, payload: Dict[str, object]) -> None:
        with self._condition:
            self._results[frame_id] = payload
            if len(self._results) > 120:
                for key in sorted(self._results)[:20]:
                    self._results.pop(key, None)
            self._condition.notify_all()

    def wait_for(
        self,
        frame_id: int,
        timeout: float,
        *,
        tick_callback=None,
        tick_hz: float = 20.0,
    ) -> Optional[Dict[str, object]]:
        deadline = time.time() + float(timeout)
        interval = 1.0 / max(1.0, float(tick_hz))
        while True:
            with self._condition:
                result = self._results.pop(frame_id, None)
                if result is not None:
                    return result
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                self._condition.wait(min(remaining, interval))
            if tick_callback is not None:
                tick_callback()


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
        self._file.flush()

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


def render_metrics_axes(
    axes: Tuple[object, object, object],
    records: List[Dict[str, object]],
    title: str,
) -> None:
    latency_ax, payload_ax, mask_ax = axes
    for axis in axes:
        axis.clear()
        axis.grid(True, alpha=0.3)

    latency_ax.set_title(title)
    latency_ax.set_ylabel("Latency (ms)")
    payload_ax.set_ylabel("Payload (KiB)")
    mask_ax.set_ylabel("Foreground px")
    mask_ax.set_xlabel("Elapsed time (s)")
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
    foreground_pixels = [float(record["mask_foreground_pixels"]) for record in records]

    latency_ax.plot(elapsed, front_ms, label="Front half", color="tab:blue", linewidth=1.8)
    latency_ax.plot(elapsed, back_ms, label="Back half", color="tab:orange", linewidth=1.8)
    latency_ax.plot(elapsed, round_trip_ms, label="Round trip", color="tab:red", linewidth=1.8)
    latency_ax.legend(loc="upper right")
    payload_ax.plot(elapsed, payload_kib, label="Compressed", color="tab:green", linewidth=1.8)
    payload_ax.plot(
        elapsed,
        payload_uncompressed_kib,
        label="Float16 baseline",
        color="tab:gray",
        linewidth=1.4,
        linestyle="--",
    )
    payload_ax.legend(loc="upper right")
    mask_ax.plot(elapsed, foreground_pixels, color="tab:purple", linewidth=1.8)


class LiveMetricsPlotter:
    def __init__(self, history_limit: int, enable_window: bool) -> None:
        self.history_limit = max(10, int(history_limit))
        self.records: List[Dict[str, object]] = []
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
                self._figure.canvas.manager.set_window_title(
                    "CARLA Split Segmentation Metrics"
                )
            except Exception:
                pass
            render_metrics_axes(
                self._axes,
                self.records,
                title="CARLA Split Segmentation Metrics (Live)",
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
        render_metrics_axes(
            self._axes,
            self.records,
            title="CARLA Split Segmentation Metrics (Live)",
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


class ManualDriveController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vehicle = None
        self.control = carla.VehicleControl()
        self._pygame = None
        self._clock = None
        self._display = None
        self._font = None
        self._initialized = False
        self._init_failed = False
        self._autopilot_enabled = False
        self._steer_cache = 0.0
        self._help_printed = False

    def set_vehicle(self, vehicle: "carla.Vehicle") -> None:
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self._autopilot_enabled = False
        try:
            vehicle.set_autopilot(False, int(self.args.tm_port))
            vehicle.apply_control(self.control)
        except RuntimeError as exc:
            print(f"Warning: unable to initialize manual vehicle control: {exc}", file=sys.stderr)
        self.print_help()

    def print_help(self) -> None:
        if self._help_printed:
            return
        self._help_printed = True
        print(
            "Manual drive controls: focus the 'CARLA Manual Drive' window. "
            "W/Up throttle, S/Down brake, A/Left and D/Right steer, Q reverse, "
            "Space hand brake, P autopilot toggle, Esc or Ctrl+Q quit."
        )

    def _ensure_pygame(self) -> bool:
        if self._initialized:
            return True
        if self._init_failed:
            return False
        if bool(getattr(self.args, "headless", False)):
            print(
                "Warning: --drive-mode manual requires a pygame control window; "
                "--headless disables manual keyboard control.",
                file=sys.stderr,
            )
            self._init_failed = True
            return False
        try:
            import pygame  # type: ignore
        except ImportError:
            print(
                "Warning: pygame is required for manual keyboard control. "
                "Install pygame or use --drive-mode autopilot.",
                file=sys.stderr,
            )
            self._init_failed = True
            return False
        try:
            pygame.init()
            pygame.font.init()
            self._display = pygame.display.set_mode((560, 160))
            pygame.display.set_caption("CARLA Manual Drive")
            self._font = pygame.font.Font(pygame.font.get_default_font(), 18)
            self._clock = pygame.time.Clock()
        except Exception as exc:
            print(f"Warning: unable to initialize pygame manual controls: {exc}", file=sys.stderr)
            self._init_failed = True
            return False
        self._pygame = pygame
        self._initialized = True
        return True

    def shutdown(self) -> None:
        if self._pygame is not None:
            try:
                self._pygame.quit()
            except Exception:
                pass
        self._pygame = None
        self._clock = None
        self._display = None
        self._font = None
        self._initialized = False

    def _render_status(self) -> None:
        if self._display is None or self._font is None or self._pygame is None:
            return
        pygame = self._pygame
        self._display.fill((18, 18, 18))
        lines = [
            "Focus this window for manual driving. Esc or Ctrl+Q quits.",
            "W/S throttle/brake | A/D steer | Q reverse | Space hand brake | P autopilot",
            (
                f"throttle={self.control.throttle:.2f} brake={self.control.brake:.2f} "
                f"steer={self.control.steer:.2f} reverse={bool(self.control.reverse)} "
                f"autopilot={self._autopilot_enabled}"
            ),
        ]
        for row, text in enumerate(lines):
            surface = self._font.render(text, True, (235, 235, 235))
            self._display.blit(surface, (14, 18 + row * 40))
        pygame.display.flip()

    def _handle_keyup(self, key: int) -> bool:
        pygame = self._pygame
        if pygame is None:
            return False
        mods = pygame.key.get_mods()
        if key == pygame.K_ESCAPE or (key == pygame.K_q and mods & pygame.KMOD_CTRL):
            return True
        if key == pygame.K_q:
            self.control.gear = 1 if bool(self.control.reverse) else -1
            self.control.reverse = self.control.gear < 0
            print(f"Manual drive reverse={'on' if self.control.reverse else 'off'}.")
        elif key == pygame.K_p:
            self._autopilot_enabled = not self._autopilot_enabled
            if self.vehicle is not None:
                self.vehicle.set_autopilot(self._autopilot_enabled, int(self.args.tm_port))
            print(f"Manual drive autopilot={'on' if self._autopilot_enabled else 'off'}.")
        return False

    def _parse_vehicle_keys(self, keys: Sequence[bool], milliseconds: int) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        throttle_step = min(1.0, max(0.0, float(self.args.manual_throttle)))
        brake_step = min(1.0, max(0.0, float(self.args.manual_brake)))
        steer_rate = min(1.0, max(0.0, float(self.args.manual_steer_step)))

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.control.throttle = min(float(self.control.throttle) + throttle_step, 1.0)
        else:
            self.control.throttle = 0.0

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.control.brake = min(float(self.control.brake) + brake_step, 1.0)
        else:
            self.control.brake = 0.0

        steer_increment = steer_rate * float(milliseconds)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self._steer_cache = min(0.0, self._steer_cache) - steer_increment
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self._steer_cache = max(0.0, self._steer_cache) + steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.control.steer = round(self._steer_cache, 1)
        self.control.hand_brake = bool(keys[pygame.K_SPACE])
        self.control.reverse = int(self.control.gear) < 0

    def tick(self) -> bool:
        if str(getattr(self.args, "drive_mode", "autopilot")) != "manual":
            return False
        if self.vehicle is None:
            return False
        if not self._ensure_pygame():
            return False
        pygame = self._pygame
        if pygame is None:
            return False

        fps = max(1, int(float(getattr(self.args, "fps", 20.0))))
        milliseconds = int(self._clock.tick(fps) if self._clock is not None else 1000.0 / fps)
        if milliseconds <= 0:
            milliseconds = int(1000.0 / fps)

        should_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            elif event.type == pygame.KEYUP:
                should_quit = self._handle_keyup(event.key) or should_quit
        if should_quit:
            return True

        if not self._autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), milliseconds)
            try:
                self.vehicle.apply_control(self.control)
            except RuntimeError:
                self.vehicle = None
        self._render_status()
        return False


class CameraSideSegmentationSplitInference:
    def __init__(
        self,
        model: TorchvisionSegmentationSplitModel,
        sender: "od_collect.UDPMessageSocket",
        *,
        transport: "od_collect.TransportConfig",
        autoencoder: Optional[PerLevelFeatureAutoencoder] = None,
        per_level_compress_probe: bool = False,
    ) -> None:
        self.model = model
        self.sender = sender
        # Codecs are dispatched by quantization mode at serialization time, so we
        # share the OD module's per-name codec registry directly.
        self.feature_codecs: Dict[str, object] = OrderedDict()
        self.transport = transport
        self.autoencoder = autoencoder
        self.per_level_compress_probe = bool(per_level_compress_probe)
        # Cache the entropy coder used by the per-level probe so we don't
        # rebuild a fresh zstd context every frame.
        self._probe_coder = transport.make_entropy_coder()

    def process(self, frame_id: int, frame_bgr: np.ndarray) -> Dict[str, object]:
        output_size = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))
        started = time.perf_counter()
        with torch.inference_mode():
            image_tensor = self.model.preprocess(frame_bgr)
            features = self.model.encode(image_tensor)

            # Saliency-based ROI gate (LR-ASPP analogue of the OD RPN-objectness gate).
            roi_masks, drop_fraction_total, drop_fraction_per_level = saliency_drop_masks(
                features, self.transport.roi_objectness_threshold
            )
            if roi_masks:
                gated: "OrderedDict[str, torch.Tensor]" = OrderedDict()
                for name, tensor in features.items():
                    mask = roi_masks[name].to(tensor.device, dtype=tensor.dtype)
                    gated[name] = tensor * mask.unsqueeze(1)
                features = gated

            ae_output_sizes: Dict[str, Tuple[int, int]] = {}
            ae_original_channels: Dict[str, int] = {}
            if self.autoencoder is not None:
                bottlenecked: "OrderedDict[str, torch.Tensor]" = OrderedDict()
                for name, tensor in features.items():
                    ae_output_sizes[name] = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                    ae_original_channels[name] = int(tensor.shape[1])
                    bottlenecked[name] = self.autoencoder.encode(name, tensor)
                features = bottlenecked

        (
            serialized_features,
            payload_bytes_uncompressed,
            per_level_uncompressed,
            per_level_compressed,
        ) = od_collect.serialize_feature_maps(
            features,
            self.feature_codecs,
            quantization_mode=self.transport.quantization_mode,
            per_level_compress_probe=self.per_level_compress_probe,
            entropy_coder=self._probe_coder,
        )
        payload = {
            "frame_id": int(frame_id),
            "batch_size": int(image_tensor.shape[0]),
            "input_size": [int(image_tensor.shape[-2]), int(image_tensor.shape[-1])],
            "output_size": [int(output_size[0]), int(output_size[1])],
            "feature_shapes": {
                name: tuple(int(v) for v in tensor.shape) for name, tensor in features.items()
            },
            "ae_output_sizes": {k: list(v) for k, v in ae_output_sizes.items()},
            "ae_original_channels": dict(ae_original_channels),
            "features": serialized_features,
            "camera_sent_perf": time.perf_counter(),
        }
        payload_bytes, payload_chunks = self.sender.send(payload)
        return {
            "front_ms": (time.perf_counter() - started) * 1000.0,
            "payload_bytes": int(payload_bytes),
            "payload_bytes_uncompressed": int(payload_bytes_uncompressed),
            "payload_chunks": int(payload_chunks),
            "roi_drop_fraction_total": float(drop_fraction_total),
            "roi_drop_fraction_per_level": dict(drop_fraction_per_level),
            "per_level_uncompressed_bytes": dict(per_level_uncompressed),
            "per_level_compressed_bytes": dict(per_level_compressed),
        }


class SegmentationRemoteInferenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        model: TorchvisionSegmentationSplitModel,
        receiver: "od_collect.UDPMessageSocket",
        sender: "od_collect.UDPMessageSocket",
        device: torch.device,
        stop_event: threading.Event,
        transport: "od_collect.TransportConfig",
        autoencoder: Optional[PerLevelFeatureAutoencoder] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.receiver = receiver
        self.sender = sender
        self.device = device
        self.stop_event = stop_event
        self.feature_codecs: Dict[str, object] = OrderedDict()
        self.transport = transport
        self.autoencoder = autoencoder

    def _run_back_half(self, payload: Dict[str, object]) -> Dict[str, object]:
        started = time.perf_counter()
        features = od_collect.deserialize_feature_maps(
            payload["features"],
            self.device,
            batch_size=int(payload.get("batch_size", 1)),
            feature_codecs=self.feature_codecs,
            quantization_mode=self.transport.quantization_mode,
        )
        if self.autoencoder is not None:
            ae_output_sizes = payload.get("ae_output_sizes") or {}
            ae_original_channels = payload.get("ae_original_channels") or {}
            decoded: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for name, tensor in features.items():
                size_hint = ae_output_sizes.get(name)
                output_size = tuple(int(v) for v in size_hint) if size_hint else None
                original_channels = int(
                    ae_original_channels.get(name, tensor.shape[1])
                )
                decoded[name] = self.autoencoder.decode(
                    name,
                    tensor,
                    original_channels=original_channels,
                    output_size=output_size,
                )
            features = decoded
        output_size = tuple(int(v) for v in payload["output_size"])
        with torch.inference_mode():
            mask = self.model.decode_mask(features, output_size=output_size)
        raw_mask = pickle.dumps(mask, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "frame_id": int(payload["frame_id"]),
            "camera_sent_perf": float(payload["camera_sent_perf"]),
            "server_ms": (time.perf_counter() - started) * 1000.0,
            "mask": mask,
            "mask_payload_bytes_estimate": len(raw_mask),
        }

    def run(self) -> None:
        while not self.stop_event.is_set():
            payload = self.receiver.receive()
            if payload is None:
                continue
            try:
                result = self._run_back_half(payload)
                self.sender.send(result)
            except Exception as exc:
                print(f"Segmentation remote worker error: {exc}", file=sys.stderr)


class CameraResultReceiver(threading.Thread):
    def __init__(
        self,
        receiver: "od_collect.UDPMessageSocket",
        result_store: SegmentationResultStore,
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


def resolve_metrics_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_dir = Path(args.metrics_log_dir).expanduser().resolve()
    prefix = str(args.metrics_log_prefix or "split_segmentation_metrics").strip()
    run_tag = str(args.run_tag or "").strip()
    filename = f"{prefix}_{run_tag}" if run_tag else prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = output_dir / f"{filename}_{timestamp}"
    return base_path.with_suffix(".csv"), base_path.with_suffix(".manifest.json")


def _parse_manifest_extra(raw_json: str) -> Dict[str, object]:
    if not raw_json:
        return {}
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        print(f"Warning: --manifest-extra-json is not valid JSON ({exc}); ignoring.", file=sys.stderr)
        return {}
    if not isinstance(parsed, dict):
        print(
            "Warning: --manifest-extra-json must decode to an object; "
            f"got {type(parsed).__name__}; ignoring.",
            file=sys.stderr,
        )
        return {}
    return parsed


def write_run_manifest(
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
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "wall_time_iso": datetime.now().isoformat(timespec="seconds"),
        "run_tag": args.run_tag or "",
        "csv_path": str(csv_path),
        "town_requested": args.town,
        "town_loaded": town_loaded,
        "weather_preset_requested": args.weather_preset,
        "weather_preset_applied": weather_applied or WEATHER_PRESET_NONE,
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
        "seg_input_width": int(seg_input_size[0]),
        "seg_input_height": int(seg_input_size[1]),
        "seg_mask_strength": float(args.seg_mask_strength),
        "metrics_warmup_frames": int(args.metrics_warmup_frames),
        "max_frames": int(args.max_frames),
        "run_duration_s": float(args.run_duration_s),
        "vehicle_blueprint": args.vehicle_blueprint,
        "drive_mode": str(args.drive_mode),
        "host": args.host,
        "port": int(args.port),
        # Transport-axis tags so the downstream analyzer can group rows the
        # same way it groups OD rows.
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
        "extra": _parse_manifest_extra(args.manifest_extra_json),
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _quality_metric_columns(
    pred_voc_mask: Optional[np.ndarray],
    gt_3class: Optional[np.ndarray],
) -> Dict[str, object]:
    """Compute the per-frame mIoU columns; NaN-safe when either side is missing.

    `pred_voc_mask` is the LR-ASPP argmax in VOC label space (uint8). It gets
    mapped to the 3-class scheme before mIoU; the GT is assumed to already be
    in the 3-class scheme (caller maps via map_carla_tags_to_3class).
    """
    columns: Dict[str, object] = {
        "gt_camera_available": int(gt_3class is not None),
        "miou_binary": float("nan"),
        "miou_3class_macro": float("nan"),
        "miou_vehicle_iou": float("nan"),
        "miou_person_iou": float("nan"),
        "gt_vehicle_pixels": 0,
        "gt_person_pixels": 0,
    }
    if pred_voc_mask is None or gt_3class is None:
        return columns
    pred_3class = map_voc_labels_to_3class(pred_voc_mask.astype(np.uint8, copy=False))
    vehicle_iou, person_iou, macro_iou, binary_iou = compute_3class_iou(
        pred_3class, gt_3class
    )
    columns["miou_binary"] = float(binary_iou)
    columns["miou_3class_macro"] = float(macro_iou)
    columns["miou_vehicle_iou"] = float(vehicle_iou)
    columns["miou_person_iou"] = float(person_iou)
    columns["gt_vehicle_pixels"] = int(np.count_nonzero(gt_3class == CLASS_ID_VEHICLE))
    columns["gt_person_pixels"] = int(np.count_nonzero(gt_3class == CLASS_ID_PERSON))
    return columns


def build_metrics_record(
    *,
    frame_id: int,
    elapsed_s: float,
    args: argparse.Namespace,
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    mask: Optional[np.ndarray],
    camera_width: int,
    camera_height: int,
    camera_resolution_label: str,
    seg_input_size: Tuple[int, int],
    town: str,
    weather_preset: str,
    gt_3class: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    back_ms = float("nan")
    round_trip_ms = float("nan")
    if remote_stats is not None:
        back_ms = float(remote_stats["server_ms"])
        round_trip_ms = float(remote_stats["round_trip_ms"])

    payload_bytes = int(front_stats["payload_bytes"])
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    mask_classes = ""
    foreground_pixels = 0
    if mask is not None:
        labels, counts = np.unique(mask, return_counts=True)
        foreground = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0]
        foreground_pixels = int(sum(count for _, count in foreground))
        mask_classes = ";".join(
            f"{VOC_SEGMENTATION_LABELS[label] if label < len(VOC_SEGMENTATION_LABELS) else label}:{count}"
            for label, count in foreground
        )

    return {
        "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_s": float(elapsed_s),
        "frame_id": int(frame_id),
        "run_tag": str(args.run_tag or ""),
        "town": str(town),
        "weather_preset": str(weather_preset or WEATHER_PRESET_NONE),
        "camera_resolution_label": str(camera_resolution_label),
        "camera_width": int(camera_width),
        "camera_height": int(camera_height),
        "camera_fov": float(args.camera_fov),
        "npc_vehicles": int(args.npc_vehicles),
        "npc_pedestrians": int(args.npc_pedestrians),
        "segmentation_model": str(args.segmentation_model),
        "seg_input_width": int(seg_input_size[0]),
        "seg_input_height": int(seg_input_size[1]),
        "front_ms": float(front_stats["front_ms"]),
        "back_ms": back_ms,
        "round_trip_ms": round_trip_ms,
        "payload_bytes": payload_bytes,
        "payload_bytes_uncompressed": payload_bytes_uncompressed,
        "payload_kib": payload_bytes / 1024.0,
        "payload_uncompressed_kib": payload_bytes_uncompressed / 1024.0,
        "payload_chunks": int(front_stats["payload_chunks"]),
        "mask_available": int(mask is not None),
        "mask_foreground_pixels": foreground_pixels,
        "mask_classes": mask_classes,
        "quantization_mode": str(args.quantization_mode),
        "entropy_coder": str(args.entropy_coder),
        "zstd_level": int(args.zstd_level),
        "roi_objectness_threshold": float(args.roi_objectness_threshold),
        "roi_drop_fraction_total": float(front_stats.get("roi_drop_fraction_total", 0.0)),
        "roi_drop_fraction_per_level_json": json.dumps(
            front_stats.get("roi_drop_fraction_per_level", {}), sort_keys=True
        ),
        "ae_mode": str(args.ae_mode),
        "ae_bottleneck_channels": int(args.ae_bottleneck_channels),
        "ae_spatial_stride": int(args.ae_spatial_stride),
        **_quality_metric_columns(mask, gt_3class),
        "per_level_uncompressed_bytes_json": json.dumps(
            front_stats.get("per_level_uncompressed_bytes", {}), sort_keys=True
        ),
        "per_level_compressed_bytes_json": json.dumps(
            front_stats.get("per_level_compressed_bytes", {}), sort_keys=True
        ),
    }


def spawn_hero_vehicle(
    args: argparse.Namespace,
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    manual_controller: Optional[ManualDriveController],
) -> "carla.Vehicle":
    if str(args.drive_mode) != "manual":
        return od_demo.spawn_hero_vehicle(client, world, traffic_manager, args.vehicle_blueprint)

    preferred, fell_back = od_demo.resolve_hero_blueprint(world, args.vehicle_blueprint)
    if fell_back:
        print(
            f"Requested blueprint {args.vehicle_blueprint!r} was not found. "
            f"Falling back to {preferred.id!r}."
        )
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        command = carla.command.SpawnActor(
            od_demo.get_fresh_vehicle_blueprint(world, preferred.id, "hero"),
            spawn_point,
        )
        response = client.apply_batch_sync([command], True)[0]
        if response.error:
            continue
        actor = world.get_actor(response.actor_id)
        if actor is None:
            continue
        actor.set_autopilot(False, int(args.tm_port))
        if manual_controller is not None:
            manual_controller.set_vehicle(actor)
        return actor
    raise RuntimeError("Unable to spawn manual hero vehicle at any spawn point.")


def _manual_tick(controller: Optional[ManualDriveController]) -> None:
    if controller is not None and controller.tick():
        raise KeyboardInterrupt


def _segmentation_mask_boundaries(mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(mask.shape[:2], dtype=bool)
    boundaries[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundaries[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundaries[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundaries[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundaries & (mask > 0)


def _draw_overlay_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font_scale: float,
    thickness: int,
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


def _mask_class_summary(mask: np.ndarray, max_items: int = 4) -> str:
    labels, counts = np.unique(mask, return_counts=True)
    foreground = sorted(
        ((int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0),
        key=lambda item: item[1],
        reverse=True,
    )
    if not foreground:
        return "classes: background"
    parts = []
    for label, count in foreground[:max_items]:
        name = VOC_SEGMENTATION_LABELS[label] if label < len(VOC_SEGMENTATION_LABELS) else str(label)
        parts.append(f"{name} {count / mask.size * 100.0:.1f}%")
    return "classes: " + ", ".join(parts)


def draw_segmentation_overlay(
    frame_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    *,
    args: argparse.Namespace,
    metrics_warmup_remaining: int = 0,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    if mask is not None:
        if mask.shape[:2] != annotated.shape[:2]:
            mask = cv2.resize(
                mask,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        colors_rgb = SEGMENTATION_OVERLAY_PALETTE_RGB[
            mask.clip(0, len(SEGMENTATION_OVERLAY_PALETTE_RGB) - 1)
        ]
        colors_bgr = colors_rgb[:, :, ::-1]
        foreground = mask > 0
        strength = min(1.0, max(0.0, float(args.seg_mask_strength)))
        annotated[foreground] = (
            annotated[foreground].astype(np.float32) * (1.0 - strength)
            + colors_bgr[foreground].astype(np.float32) * strength
        ).astype(np.uint8)

        boundaries = _segmentation_mask_boundaries(mask)
        if boundaries.any():
            outline = cv2.dilate(
                boundaries.astype(np.uint8),
                SEGMENTATION_BOUNDARY_KERNEL,
                iterations=1,
            ).astype(bool)
            outline &= foreground
            annotated[outline] = (0, 0, 0)
            annotated[boundaries] = (255, 255, 255)

    payload_bytes = max(1, int(front_stats["payload_bytes"]))
    payload_bytes_uncompressed = int(front_stats["payload_bytes_uncompressed"])
    compression_ratio = payload_bytes_uncompressed / payload_bytes
    lines = [
        "LR-ASPP split segmentation",
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
        _mask_class_summary(mask) if mask is not None else "mask: waiting",
        f"drive: {args.drive_mode}",
    ]
    if remote_stats is not None:
        lines.append(f"Back half: {remote_stats['server_ms']:.1f} ms")
        lines.append(f"Round trip: {remote_stats['round_trip_ms']:.1f} ms")
    if metrics_warmup_remaining > 0:
        lines.append(f"Metrics warm-up: {metrics_warmup_remaining} frame(s) remaining")

    y = 28
    for line in lines:
        _draw_overlay_text(
            annotated,
            line,
            (10, y),
            font_scale=0.56,
            thickness=2,
        )
        y += 24
    return annotated


def run_demo(args: argparse.Namespace) -> None:
    random.seed(7)
    front_device = od_demo.resolve_device(args.front_device)
    back_device = od_demo.resolve_device(args.back_device)
    camera_width, camera_height, camera_resolution_label = od_demo.resolve_camera_dimensions(args)
    seg_input_size = _resolve_seg_input_size(args, camera_width, camera_height)
    metrics_csv_path: Optional[Path] = None
    metrics_manifest_path: Optional[Path] = None
    if args.collect_metrics:
        metrics_csv_path, metrics_manifest_path = resolve_metrics_output_paths(args)
    gui_enabled = od_demo.has_graphical_display() and not args.headless

    if front_device.type == "cuda" or back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    front_raw_model, back_raw_model = build_segmentation_models(args)
    front_model = TorchvisionSegmentationSplitModel(
        front_raw_model,
        front_device,
        input_size=seg_input_size,
    )
    back_model = TorchvisionSegmentationSplitModel(
        back_raw_model,
        back_device,
        input_size=seg_input_size,
    )

    transport_cfg = od_collect.TransportConfig(
        quantization_mode=str(args.quantization_mode),
        entropy_coder_name=str(args.entropy_coder),
        zstd_level=int(args.zstd_level),
        roi_objectness_threshold=float(args.roi_objectness_threshold),
        # The RCNN-transform bypass is OD-specific (LR-ASPP has no
        # GeneralizedRCNNTransform); always False here. We still reuse the
        # shared TransportConfig dataclass so analysis tooling stays uniform.
        bypass_rcnn_transform=False,
    )
    front_autoencoder = build_per_level_autoencoder(args, front_device)
    back_autoencoder = build_per_level_autoencoder(args, back_device)

    # Each socket gets its own coder so send/receive paths don't share a single
    # zstd context across threads. Same idiom as the OD data-collect demo.
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
    result_store = SegmentationResultStore()
    split_camera = CameraSideSegmentationSplitInference(
        front_model,
        camera_sender,
        transport=transport_cfg,
        autoencoder=front_autoencoder,
        per_level_compress_probe=bool(args.per_level_compress_probe),
    )
    remote_worker = SegmentationRemoteInferenceWorker(
        model=back_model,
        receiver=remote_receiver,
        sender=remote_sender,
        device=back_device,
        stop_event=stop_event,
        transport=transport_cfg,
        autoencoder=back_autoencoder,
    )
    result_receiver = CameraResultReceiver(
        receiver=camera_receiver,
        result_store=result_store,
        stop_event=stop_event,
    )
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
    weather_applied = apply_weather_preset(world, args.weather_preset)

    actors: List["carla.Actor"] = []
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
    metrics_collector = None
    if metrics_csv_path is not None or bool(args.live_plot):
        metrics_collector = AsyncMetricsCollector(
            csv_path=metrics_csv_path,
            enable_live_plot=bool(args.live_plot),
            gui_enabled=gui_enabled,
            args=args,
        )
        metrics_collector.start()
    manual_controller = ManualDriveController(args) if args.drive_mode == "manual" else None

    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"Town: {world.get_map().name}")
    print(f"Weather: {weather_applied or WEATHER_PRESET_NONE}")
    print(f"Segmentation model: {args.segmentation_model}, input={seg_input_size[0]}x{seg_input_size[1]}")
    print(f"Front device: {front_device}, back device: {back_device}")
    print(f"Camera resolution: {camera_width}x{camera_height} ({camera_resolution_label})")
    print(f"Drive mode: {args.drive_mode}")
    if metrics_csv_path is not None:
        print(f"Metrics CSV: {metrics_csv_path}")
    else:
        print("Metrics data collection disabled. CSV logging is off.")
    if metrics_collector is not None and metrics_collector.warning:
        print(metrics_collector.warning)
    if not gui_enabled:
        if args.headless:
            print("GUI disabled by --headless. Running without the OpenCV view.")
        else:
            print("No graphical display detected. Running without the OpenCV view.")
    print(
        "UDP ports: "
        f"camera {args.camera_source_port} -> remote {args.remote_port}, "
        f"remote {args.remote_source_port} -> camera {args.camera_result_port}"
    )

    try:
        hero_vehicle = spawn_hero_vehicle(args, client, world, traffic_manager, manual_controller)
        actors.append(hero_vehicle)
        print(f"Hero vehicle: {hero_vehicle.type_id}")

        background_vehicles = od_demo.spawn_background_traffic(
            client,
            world,
            traffic_manager,
            args.npc_vehicles,
            hero_vehicle,
        )
        actors.extend(background_vehicles)
        if background_vehicles:
            print(f"Spawned {len(background_vehicles)} background vehicles.")

        pedestrian_walkers, pedestrian_controllers = od_demo.spawn_background_pedestrians(
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
        camera_transform = carla.Transform(carla.Location(x=args.camera_x, z=args.camera_z))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=hero_vehicle)
        actors.append(camera)
        camera.listen(lambda image: od_demo.put_latest(image_queue, image))
        first_image = od_demo.warmup_camera_stream(
            world,
            image_queue,
            args.camera_warmup_ticks,
            args.camera_timeout,
        )
        print(f"Camera ready on frame {first_image.frame}.")

        gt_queue: Optional["queue.Queue[carla.Image]"] = None
        gt_camera = None
        if bool(args.enable_semantic_gt):
            gt_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
            gt_bp.set_attribute("image_size_x", str(camera_width))
            gt_bp.set_attribute("image_size_y", str(camera_height))
            gt_bp.set_attribute("fov", str(args.camera_fov))
            gt_bp.set_attribute("sensor_tick", str(1.0 / args.fps))
            gt_camera = world.spawn_actor(gt_bp, camera_transform, attach_to=hero_vehicle)
            actors.append(gt_camera)
            gt_queue = queue.Queue(maxsize=2)
            gt_camera.listen(lambda image, q=gt_queue: od_demo.put_latest(q, image))
            print("Semantic GT camera enabled (3-class mIoU logging is on).")

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

        if gui_enabled:
            cv2.namedWindow("CARLA Split Segmentation", cv2.WINDOW_AUTOSIZE)
        else:
            print("Headless run active. Press Ctrl+C to stop the demo.")

        while True:
            _manual_tick(manual_controller)
            world_frame = int(world.tick())
            image = od_demo.wait_for_camera_frame(image_queue, world_frame, args.camera_timeout)
            if image is None:
                print(
                    f"Warning: camera frame for world tick {world_frame} was not received "
                    f"within {args.camera_timeout:.1f}s; retrying."
                )
                continue

            frame_bgr = od_demo.camera_image_to_bgr(image)
            front_stats = split_camera.process(int(image.frame), frame_bgr)

            gt_3class: Optional[np.ndarray] = None
            if gt_queue is not None:
                gt_image = od_demo.wait_for_camera_frame(
                    gt_queue, world_frame, args.camera_timeout
                )
                if gt_image is not None:
                    gt_tags = carla_semantic_image_to_tags(gt_image)
                    gt_3class = map_carla_tags_to_3class(gt_tags)
                else:
                    print(
                        f"Warning: semantic-GT frame for tick {world_frame} was not "
                        f"received within {args.camera_timeout:.1f}s; mIoU columns "
                        "will be NaN for this frame."
                    )
            result = result_store.wait_for(
                int(image.frame),
                args.result_timeout,
                tick_callback=lambda: _manual_tick(manual_controller),
                tick_hz=float(args.fps),
            )

            remote_stats = None
            mask: Optional[np.ndarray] = None
            if result is not None:
                remote_stats = {
                    "server_ms": float(result["server_ms"]),
                    "round_trip_ms": (time.perf_counter() - float(result["camera_sent_perf"])) * 1000.0,
                }
                mask = result.get("mask") if isinstance(result.get("mask"), np.ndarray) else None

            if needs_measurement_window:
                if metrics_warmup_remaining > 0:
                    metrics_warmup_remaining -= 1
                    if metrics_warmup_remaining == 0:
                        metrics_start_perf = time.perf_counter()
                else:
                    if metrics_start_perf is None:
                        metrics_start_perf = time.perf_counter()
                    elapsed_s = time.perf_counter() - metrics_start_perf
                    if metrics_collector is not None:
                        metrics_collector.submit(
                            build_metrics_record(
                                frame_id=int(image.frame),
                                elapsed_s=elapsed_s,
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
                    if run_duration_s > 0.0 and elapsed_s >= run_duration_s:
                        print(f"Reached --run-duration-s={run_duration_s:.1f}; stopping run.")
                        break

            if gui_enabled:
                annotated = draw_segmentation_overlay(
                    frame_bgr,
                    mask,
                    front_stats,
                    remote_stats,
                    args=args,
                    metrics_warmup_remaining=metrics_warmup_remaining,
                )
                cv2.imshow("CARLA Split Segmentation", annotated)
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
        remote_receiver.close()
        remote_sender.close()
        camera_receiver.close()
        remote_worker.join(timeout=1.0)
        result_receiver.join(timeout=1.0)

        if gui_enabled:
            cv2.destroyAllWindows()
        if manual_controller is not None:
            manual_controller.shutdown()
        if metrics_collector is not None:
            metrics_collector.close()
        if metrics_csv_path is not None:
            print(f"Saved metrics CSV to {metrics_csv_path}")
        if metrics_manifest_path is not None and metrics_csv_path is not None:
            try:
                write_run_manifest(
                    metrics_manifest_path,
                    args,
                    csv_path=metrics_csv_path,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    camera_resolution_label=camera_resolution_label,
                    seg_input_size=seg_input_size,
                    weather_applied=weather_applied,
                    town_loaded=world.get_map().name,
                )
                print(f"Saved run manifest to {metrics_manifest_path}")
            except Exception as exc:
                print(f"Warning: unable to write run manifest: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.metrics_plot_worker:
        raise SystemExit(run_metrics_plot_worker(args))
    if args.drive_mode == "manual" and bool(args.headless):
        print(
            "Warning: --drive-mode manual uses a pygame control window, "
            "but --headless disables manual keyboard control."
        )
    run_demo(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")

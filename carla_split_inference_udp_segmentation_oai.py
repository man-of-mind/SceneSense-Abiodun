#!/usr/bin/env python3

"""
CARLA split semantic segmentation over the OAI 5G path.

This is the segmentation sibling of carla_split_inference_udp_oai.py.  The
front role runs CARLA, the RGB camera, the LR-ASPP backbone, GUI overlay, and
metrics.  The back role runs only the LR-ASPP classifier head and can therefore
run inside the oai-perception-rx container without importing CARLA.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import queue
import random
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import carla_split_inference_udp_oai as od_oai

from torchvision.models.segmentation import (  # noqa: E402
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)


DEFAULT_METRICS_LOG_DIR = Path(__file__).resolve().parent / "metrics_logs"
DEFAULT_WEIGHTS = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
SEGMENTATION_LABELS = DEFAULT_WEIGHTS.meta["categories"]
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
CARLA_3CLASS_SEGMENTATION_LABELS = ("background", "vehicle", "person")
CLASS_ID_BACKGROUND = 0
CLASS_ID_VEHICLE = 1
CLASS_ID_PERSON = 2
_CARLA_TAG_TO_3CLASS: Dict[int, int] = {
    12: CLASS_ID_PERSON,  # Pedestrian
    13: CLASS_ID_PERSON,  # Rider
    14: CLASS_ID_VEHICLE,  # Car
    15: CLASS_ID_VEHICLE,  # Truck
    16: CLASS_ID_VEHICLE,  # Bus
    17: CLASS_ID_VEHICLE,  # Train
    18: CLASS_ID_VEHICLE,  # Motorcycle
    19: CLASS_ID_VEHICLE,  # Bicycle
}
_VOC_LABEL_TO_3CLASS: Dict[int, int] = {
    1: CLASS_ID_VEHICLE,  # aeroplane
    2: CLASS_ID_VEHICLE,  # bicycle
    4: CLASS_ID_VEHICLE,  # boat
    6: CLASS_ID_VEHICLE,  # bus
    7: CLASS_ID_VEHICLE,  # car
    14: CLASS_ID_VEHICLE,  # motorbike
    19: CLASS_ID_VEHICLE,  # train
    15: CLASS_ID_PERSON,  # person
}
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
    "mask_available",
    "mask_payload_bytes_estimate",
    "mask_foreground_pixels",
    "mask_classes",
    "gt_camera_available",
    "miou_binary",
    "miou_3class_macro",
    "miou_vehicle_iou",
    "miou_person_iou",
    "gt_vehicle_pixels",
    "gt_person_pixels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA split segmentation over UDP/OAI.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument("--town", default="Town10HD_Opt", help="Town to load.")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port.")
    parser.add_argument(
        "--vehicle-blueprint",
        default="vehicle.lincoln.mkz_2017",
        help="Blueprint id for the hero vehicle.",
    )
    parser.add_argument(
        "--camera-resolution",
        choices=["custom", *od_oai.CAMERA_RESOLUTION_PRESETS.keys()],
        default="custom",
        help="Preset camera resolution. Use custom to honor width/height.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=384)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--camera-warmup-ticks", type=int, default=8)
    parser.add_argument("--camera-x", type=float, default=1.6)
    parser.add_argument("--camera-z", type=float, default=1.7)
    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=30)
    parser.add_argument(
        "--seg-pretrained",
        dest="seg_pretrained",
        action="store_true",
        help="Load torchvision LR-ASPP pretrained weights.",
    )
    parser.add_argument(
        "--seg-disable-pretrained",
        dest="seg_pretrained",
        action="store_false",
        help="Use random LR-ASPP weights for transport smoke tests.",
    )
    parser.set_defaults(seg_pretrained=True, collect_metrics=True)
    parser.add_argument(
        "--seg-weights-path",
        default="",
        help="Optional local LR-ASPP checkpoint path.",
    )
    parser.add_argument(
        "--seg-num-classes",
        type=int,
        default=len(SEGMENTATION_LABELS),
        help=(
            "Number of output segmentation classes. Keep 21 for torchvision "
            "VOC weights; use 3 with --seg-disable-pretrained for CARLA "
            "3-class checkpoints."
        ),
    )
    parser.add_argument(
        "--seg-class-scheme",
        choices=("voc", "carla_3class"),
        default="voc",
        help="Prediction label scheme used when computing 3-class mIoU.",
    )
    parser.add_argument(
        "--seg-input-width",
        type=int,
        default=512,
        help="LR-ASPP input width. Use 0 to match camera width.",
    )
    parser.add_argument(
        "--seg-input-height",
        type=int,
        default=288,
        help="LR-ASPP input height. Use 0 to match camera height.",
    )
    parser.add_argument(
        "--mask-output-size",
        choices=("model", "camera"),
        default="model",
        help=(
            "Size of the mask sent back by the server. 'model' keeps the return "
            "payload small; the front resizes it for display."
        ),
    )
    parser.add_argument("--seg-mask-strength", type=float, default=0.72)
    parser.add_argument("--camera-source-port", type=int, default=36100)
    parser.add_argument("--remote-port", type=int, default=36101)
    parser.add_argument("--remote-source-port", type=int, default=36102)
    parser.add_argument("--camera-result-port", type=int, default=36103)
    parser.add_argument("--chunk-bytes", type=int, default=60000)
    parser.add_argument("--result-timeout", type=float, default=0.35)
    parser.add_argument("--socket-timeout", type=float, default=0.25)
    parser.add_argument("--front-device", default="auto")
    parser.add_argument("--back-device", default="auto")
    parser.add_argument("--metrics-log-dir", default=str(DEFAULT_METRICS_LOG_DIR))
    parser.add_argument("--metrics-log-prefix", default="split_segmentation_oai_metrics")
    parser.add_argument("--metrics-warmup-frames", type=int, default=od_oai.DEFAULT_METRICS_WARMUP_FRAMES)
    parser.add_argument("--headless", action="store_true", help="Disable the OpenCV view.")
    parser.add_argument(
        "--role",
        choices=("loopback", "front", "back"),
        default="loopback",
        help="'loopback' runs both halves, 'front' runs CARLA/backbone, 'back' runs classifier only.",
    )
    parser.add_argument(
        "--bind-host",
        default=od_oai.DEFAULT_HOST,
        help="Local interface for UDP binds, e.g. 10.0.0.2 on the UE side.",
    )
    parser.add_argument(
        "--remote-host",
        default=None,
        help="Peer IP, e.g. 192.168.70.140 on front or 10.0.0.2 on back.",
    )
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--enable-metrics-collection",
        "--enable-data-collection",
        dest="collect_metrics",
        action="store_true",
    )
    metrics_group.add_argument(
        "--disable-metrics-collection",
        "--disable-data-collection",
        dest="collect_metrics",
        action="store_false",
    )
    semantic_gt_group = parser.add_mutually_exclusive_group()
    semantic_gt_group.add_argument(
        "--enable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_true",
        help=(
            "Spawn a co-located CARLA semantic-segmentation camera on the "
            "front side and log binary, macro, vehicle, and person IoU."
        ),
    )
    semantic_gt_group.add_argument(
        "--disable-semantic-gt",
        dest="enable_semantic_gt",
        action="store_false",
        help="Do not spawn the evaluation-only semantic GT camera.",
    )
    parser.set_defaults(enable_semantic_gt=False)
    return parser.parse_args()


def resolve_seg_input_size(
    args: argparse.Namespace,
    camera_width: int,
    camera_height: int,
) -> Tuple[int, int]:
    width = int(args.seg_input_width) if int(args.seg_input_width) > 0 else int(camera_width)
    height = int(args.seg_input_height) if int(args.seg_input_height) > 0 else int(camera_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid segmentation input size {width}x{height}")
    return width, height


def resolve_metrics_csv_path(args: argparse.Namespace) -> Path:
    output_dir = Path(args.metrics_log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.metrics_log_prefix or "split_segmentation_oai_metrics").strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{timestamp}.csv"


class MetricsCSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRICS_CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self.sample_count = 0

    def append(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self.sample_count += 1
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def load_state_dict_if_requested(model: torch.nn.Module, path: str) -> None:
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
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Warning: segmentation checkpoint keys did not exactly match.")
        if incompatible.missing_keys:
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")


def build_raw_lraspp_model(args: argparse.Namespace) -> torch.nn.Module:
    if bool(args.seg_pretrained):
        try:
            model = lraspp_mobilenet_v3_large(weights=DEFAULT_WEIGHTS)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load LR-ASPP pretrained weights. Copy the weights into "
                "TORCH_HOME, run with internet access, pass --seg-weights-path, "
                "or use --seg-disable-pretrained for a transport smoke test."
            ) from exc
    else:
        model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=int(args.seg_num_classes),
        )
    load_state_dict_if_requested(model, str(args.seg_weights_path or ""))
    return model.eval()


def clone_segmentation_model(
    reference_model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.nn.Module:
    clone_args = argparse.Namespace(
        seg_pretrained=False,
        seg_weights_path="",
        seg_num_classes=int(args.seg_num_classes),
    )
    cloned = build_raw_lraspp_model(clone_args)
    cloned.load_state_dict(reference_model.state_dict())
    cloned.eval()
    return cloned


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
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        logits = self.model.classifier(features)
        if isinstance(logits, dict):
            logits = logits["out"]
        input_width, input_height = int(input_size[0]), int(input_size[1])
        if tuple(logits.shape[-2:]) != (input_height, input_width):
            logits = F.interpolate(
                logits,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            )
        if tuple(logits.shape[-2:]) != tuple(output_size):
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
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        logits = self.decode_logits(features, input_size=input_size, output_size=output_size)
        return logits.argmax(dim=1).squeeze(0).detach().to("cpu").numpy().astype(np.uint8)


class CameraSideSegmentationSplitInference:
    def __init__(
        self,
        model: TorchvisionSegmentationSplitModel,
        sender: od_oai.UDPMessageSocket,
    ) -> None:
        self.model = model
        self.sender = sender
        self.feature_codecs: Dict[str, od_oai.SimpleFeatureCodec] = OrderedDict()

    def process(
        self,
        frame_id: int,
        frame_bgr: np.ndarray,
        *,
        mask_output_size: Tuple[int, int],
    ) -> Dict[str, object]:
        started = time.perf_counter()
        with torch.inference_mode():
            image_tensor = self.model.preprocess(frame_bgr)
            features = self.model.encode(image_tensor)
        serialized_features, payload_bytes_uncompressed = od_oai.serialize_feature_maps(
            features,
            self.feature_codecs,
        )
        payload = {
            "frame_id": int(frame_id),
            "batch_size": int(image_tensor.shape[0]),
            "input_size": [int(image_tensor.shape[-1]), int(image_tensor.shape[-2])],
            "output_size": [int(mask_output_size[0]), int(mask_output_size[1])],
            "features": serialized_features,
            "camera_sent_perf": time.perf_counter(),
        }
        payload_bytes, payload_chunks = self.sender.send(payload)
        return {
            "front_ms": (time.perf_counter() - started) * 1000.0,
            "payload_bytes": int(payload_bytes),
            "payload_bytes_uncompressed": int(payload_bytes_uncompressed),
            "payload_chunks": int(payload_chunks),
        }


class SegmentationRemoteInferenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        model: TorchvisionSegmentationSplitModel,
        receiver: od_oai.UDPMessageSocket,
        sender: od_oai.UDPMessageSocket,
        device: torch.device,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.receiver = receiver
        self.sender = sender
        self.device = device
        self.stop_event = stop_event
        self.feature_codecs: Dict[str, od_oai.SimpleFeatureCodec] = OrderedDict()

    def _run_back_half(self, payload: Dict[str, object]) -> Dict[str, object]:
        started = time.perf_counter()
        features = od_oai.deserialize_feature_maps(
            payload["features"],
            self.device,
            batch_size=int(payload.get("batch_size", 1)),
            feature_codecs=self.feature_codecs,
        )
        input_size = tuple(int(v) for v in payload["input_size"])
        output_size = tuple(int(v) for v in payload["output_size"])
        with torch.inference_mode():
            mask = self.model.decode_mask(
                features,
                input_size=input_size,
                output_size=(int(output_size[1]), int(output_size[0])),
            )
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
        receiver: od_oai.UDPMessageSocket,
        result_store: od_oai.DetectionResultStore,
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


def _build_lookup(table: Dict[int, int], size: int = 256) -> np.ndarray:
    lut = np.zeros(size, dtype=np.uint8)
    for raw, mapped in table.items():
        if 0 <= raw < size:
            lut[int(raw)] = int(mapped)
    return lut


_CARLA_LUT = _build_lookup(_CARLA_TAG_TO_3CLASS)
_VOC_LUT = _build_lookup(_VOC_LABEL_TO_3CLASS)


def segmentation_label_names(args: argparse.Namespace) -> Tuple[str, ...]:
    if str(args.seg_class_scheme) == "carla_3class":
        return CARLA_3CLASS_SEGMENTATION_LABELS
    return tuple(SEGMENTATION_LABELS)


def carla_semantic_image_to_tags(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)
    )
    return arr[:, :, 2].copy()


def map_carla_tags_to_3class(tags: np.ndarray) -> np.ndarray:
    return _CARLA_LUT[tags]


def prediction_mask_to_3class(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if str(args.seg_class_scheme) == "carla_3class":
        pred = np.zeros(mask.shape, dtype=np.uint8)
        pred[mask == CLASS_ID_VEHICLE] = CLASS_ID_VEHICLE
        pred[mask == CLASS_ID_PERSON] = CLASS_ID_PERSON
        return pred
    return _VOC_LUT[mask]


def compute_3class_iou(
    pred_3class: np.ndarray,
    gt_3class: np.ndarray,
) -> Tuple[float, float, float, float]:
    if pred_3class.shape != gt_3class.shape:
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

    pred_fg = pred_3class != CLASS_ID_BACKGROUND
    gt_fg = gt_3class != CLASS_ID_BACKGROUND
    union_fg = int(np.logical_or(pred_fg, gt_fg).sum())
    binary_iou = (
        int(np.logical_and(pred_fg, gt_fg).sum()) / union_fg
        if union_fg > 0
        else float("nan")
    )
    macro_iou = float(np.mean(ious)) if ious else float("nan")
    return (
        per_class[CLASS_ID_VEHICLE],
        per_class[CLASS_ID_PERSON],
        macro_iou,
        binary_iou,
    )


def quality_metric_columns(
    mask: Optional[np.ndarray],
    gt_3class: Optional[np.ndarray],
    args: argparse.Namespace,
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
    if mask is None or gt_3class is None:
        return columns

    pred_3class = prediction_mask_to_3class(mask.astype(np.uint8, copy=False), args)
    vehicle_iou, person_iou, macro_iou, binary_iou = compute_3class_iou(
        pred_3class,
        gt_3class,
    )
    columns["miou_binary"] = float(binary_iou)
    columns["miou_3class_macro"] = float(macro_iou)
    columns["miou_vehicle_iou"] = float(vehicle_iou)
    columns["miou_person_iou"] = float(person_iou)
    columns["gt_vehicle_pixels"] = int(np.count_nonzero(gt_3class == CLASS_ID_VEHICLE))
    columns["gt_person_pixels"] = int(np.count_nonzero(gt_3class == CLASS_ID_PERSON))
    return columns


def segmentation_mask_boundaries(mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(mask.shape[:2], dtype=bool)
    boundaries[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundaries[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundaries[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundaries[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundaries & (mask > 0)


def mask_class_summary(
    mask: Optional[np.ndarray],
    args: argparse.Namespace,
    max_items: int = 4,
) -> str:
    if mask is None:
        return "mask: waiting"
    labels, counts = np.unique(mask, return_counts=True)
    label_names = segmentation_label_names(args)
    foreground = sorted(
        ((int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0),
        key=lambda item: item[1],
        reverse=True,
    )
    if not foreground:
        return "classes: background"
    parts = []
    for label, count in foreground[:max_items]:
        name = label_names[label] if label < len(label_names) else str(label)
        parts.append(f"{name} {count / mask.size * 100.0:.1f}%")
    return "classes: " + ", ".join(parts)


def draw_text(image: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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
        boundaries = segmentation_mask_boundaries(mask)
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
            f"{float(front_stats['payload_bytes']) / 1024.0:.1f} KiB "
            f"in {front_stats['payload_chunks']} UDP chunks"
        ),
        (
            "Float16 baseline: "
            f"{payload_bytes_uncompressed / 1024.0:.1f} KiB, ratio {compression_ratio:.2f}x"
        ),
        mask_class_summary(mask, args),
    ]
    if remote_stats is not None:
        lines.append(f"Back half: {remote_stats['server_ms']:.1f} ms")
        lines.append(f"Round trip: {remote_stats['round_trip_ms']:.1f} ms")
    if metrics_warmup_remaining > 0:
        lines.append(f"Metrics warm-up: {metrics_warmup_remaining} frame(s) remaining")

    y = 28
    for line in lines:
        draw_text(annotated, line, (10, y))
        y += 24
    return annotated


def build_metrics_record(
    *,
    frame_id: int,
    elapsed_s: float,
    args: argparse.Namespace,
    front_stats: Dict[str, object],
    remote_stats: Optional[Dict[str, object]],
    mask: Optional[np.ndarray],
    gt_3class: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    back_ms = float("nan")
    round_trip_ms = float("nan")
    mask_payload_bytes_estimate = 0
    if remote_stats is not None:
        back_ms = float(remote_stats["server_ms"])
        round_trip_ms = float(remote_stats["round_trip_ms"])
        mask_payload_bytes_estimate = int(remote_stats.get("mask_payload_bytes_estimate", 0))

    mask_classes = ""
    foreground_pixels = 0
    if mask is not None:
        labels, counts = np.unique(mask, return_counts=True)
        foreground = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0]
        foreground_pixels = int(sum(count for _, count in foreground))
        label_names = segmentation_label_names(args)
        mask_classes = ";".join(
            f"{label_names[label] if label < len(label_names) else label}:{count}"
            for label, count in foreground
        )

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
        "mask_available": int(mask is not None),
        "mask_payload_bytes_estimate": mask_payload_bytes_estimate,
        "mask_foreground_pixels": foreground_pixels,
        "mask_classes": mask_classes,
        **quality_metric_columns(mask, gt_3class, args),
    }


def run_back_only(args: argparse.Namespace) -> None:
    back_device = od_oai.resolve_device(args.back_device)
    if back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    raw_model = build_raw_lraspp_model(args)
    seg_input_size = resolve_seg_input_size(args, args.camera_width, args.camera_height)
    back_model = TorchvisionSegmentationSplitModel(
        raw_model,
        back_device,
        input_size=seg_input_size,
    )
    remote_host = args.remote_host if args.remote_host is not None else args.bind_host

    remote_receiver = od_oai.UDPMessageSocket(
        bind_port=args.remote_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
    )
    remote_sender = od_oai.UDPMessageSocket(
        bind_port=args.remote_source_port,
        remote_port=args.camera_result_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
    )

    stop_event = threading.Event()
    remote_worker = SegmentationRemoteInferenceWorker(
        model=back_model,
        receiver=remote_receiver,
        sender=remote_sender,
        device=back_device,
        stop_event=stop_event,
    )
    remote_worker.start()
    print(
        f"[seg-back] device={back_device} "
        f"recv {args.bind_host}:{args.remote_port}, "
        f"send -> {remote_host}:{args.camera_result_port}"
    )
    print("[seg-back] Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[seg-back] Interrupted.")
    finally:
        stop_event.set()
        for sock in (remote_receiver, remote_sender):
            try:
                sock.close()
            except OSError:
                pass
        remote_worker.join(timeout=2.0)
        print("[seg-back] Done.")


def run_demo(args: argparse.Namespace) -> None:
    if args.role == "back":
        run_back_only(args)
        return

    carla = od_oai.ensure_carla()
    random.seed(7)

    front_device = od_oai.resolve_device(args.front_device)
    back_device = od_oai.resolve_device(args.back_device)
    camera_width, camera_height, camera_resolution_label = od_oai.resolve_camera_dimensions(args)
    seg_input_size = resolve_seg_input_size(args, camera_width, camera_height)
    gui_enabled = od_oai.has_graphical_display() and not args.headless
    metrics_csv_path = resolve_metrics_csv_path(args) if args.collect_metrics else None

    if front_device.type == "cuda" or back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    front_raw_model = build_raw_lraspp_model(args)
    back_raw_model = clone_segmentation_model(front_raw_model, args) if args.role == "loopback" else None
    front_model = TorchvisionSegmentationSplitModel(
        front_raw_model,
        front_device,
        input_size=seg_input_size,
    )
    back_model = (
        TorchvisionSegmentationSplitModel(back_raw_model, back_device, input_size=seg_input_size)
        if back_raw_model is not None
        else None
    )

    remote_host = args.remote_host if args.remote_host is not None else args.bind_host
    camera_sender = od_oai.UDPMessageSocket(
        bind_port=args.camera_source_port,
        remote_port=args.remote_port,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
        remote_host=remote_host,
    )
    remote_receiver = (
        od_oai.UDPMessageSocket(
            bind_port=args.remote_port,
            remote_port=None,
            chunk_bytes=args.chunk_bytes,
            socket_timeout=args.socket_timeout,
            host=args.bind_host,
        )
        if args.role == "loopback"
        else None
    )
    remote_sender = (
        od_oai.UDPMessageSocket(
            bind_port=args.remote_source_port,
            remote_port=args.camera_result_port,
            chunk_bytes=args.chunk_bytes,
            socket_timeout=args.socket_timeout,
            host=args.bind_host,
            remote_host=remote_host,
        )
        if args.role == "loopback"
        else None
    )
    camera_receiver = od_oai.UDPMessageSocket(
        bind_port=args.camera_result_port,
        remote_port=None,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        host=args.bind_host,
    )

    stop_event = threading.Event()
    result_store = od_oai.DetectionResultStore()
    split_camera = CameraSideSegmentationSplitInference(front_model, camera_sender)
    remote_worker = (
        SegmentationRemoteInferenceWorker(
            model=back_model,
            receiver=remote_receiver,
            sender=remote_sender,
            device=back_device,
            stop_event=stop_event,
        )
        if args.role == "loopback"
        else None
    )
    result_receiver = CameraResultReceiver(camera_receiver, result_store, stop_event)
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
    metrics_logger = MetricsCSVLogger(metrics_csv_path) if metrics_csv_path is not None else None

    print(f"Connected to CARLA at {args.host}:{args.port}")
    print(f"Town: {world.get_map().name}")
    print(f"Front device: {front_device}, back device: {back_device}")
    print(f"Camera resolution: {camera_width}x{camera_height} ({camera_resolution_label})")
    print(f"Segmentation input: {seg_input_size[0]}x{seg_input_size[1]}")
    print(f"Mask output size: {args.mask_output_size}")
    if metrics_csv_path is not None:
        print(f"Metrics CSV: {metrics_csv_path}")
    else:
        print("Metrics data collection disabled. CSV logging is off.")
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
        hero_vehicle = od_oai.spawn_hero_vehicle(
            client,
            world,
            traffic_manager,
            args.vehicle_blueprint,
        )
        actors.append(hero_vehicle)
        print(f"Hero vehicle: {hero_vehicle.type_id}")

        background_vehicles = od_oai.spawn_background_traffic(
            client,
            world,
            traffic_manager,
            args.npc_vehicles,
            hero_vehicle,
        )
        actors.extend(background_vehicles)
        if background_vehicles:
            print(f"Spawned {len(background_vehicles)} background vehicles.")

        pedestrian_walkers, pedestrian_controllers = od_oai.spawn_background_pedestrians(
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
        camera.listen(lambda image: od_oai.put_latest(image_queue, image))
        first_image = od_oai.warmup_camera_stream(
            world,
            image_queue,
            args.camera_warmup_ticks,
            args.camera_timeout,
        )
        print(f"Camera ready on frame {first_image.frame}.")

        gt_queue: Optional["queue.Queue[carla.Image]"] = None
        if bool(args.enable_semantic_gt):
            gt_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
            gt_bp.set_attribute("image_size_x", str(camera_width))
            gt_bp.set_attribute("image_size_y", str(camera_height))
            gt_bp.set_attribute("fov", str(args.camera_fov))
            gt_bp.set_attribute("sensor_tick", str(1.0 / args.fps))
            gt_camera = world.spawn_actor(gt_bp, camera_transform, attach_to=hero_vehicle)
            actors.append(gt_camera)
            gt_queue = queue.Queue(maxsize=2)
            gt_camera.listen(lambda image, q=gt_queue: od_oai.put_latest(q, image))
            print("Semantic GT camera enabled (3-class mIoU logging is on).")

        metrics_start_perf: Optional[float] = None
        metrics_warmup_remaining = (
            max(0, int(args.metrics_warmup_frames)) if metrics_logger is not None else 0
        )
        if metrics_warmup_remaining == 0:
            metrics_start_perf = time.perf_counter()
        elif metrics_logger is not None:
            print(
                "Metrics warm-up: skipping the first "
                f"{metrics_warmup_remaining} frame(s) while feature range trackers stabilize."
            )

        if gui_enabled:
            cv2.namedWindow("CARLA Split Segmentation OAI", cv2.WINDOW_AUTOSIZE)
        else:
            print("Headless run active. Press Ctrl+C to stop the demo.")

        while True:
            world_frame = int(world.tick())
            image = od_oai.wait_for_camera_frame(image_queue, world_frame, args.camera_timeout)
            if image is None:
                print(
                    f"Warning: camera frame for world tick {world_frame} was not received "
                    f"within {args.camera_timeout:.1f}s; retrying."
                )
                continue

            frame_bgr = od_oai.camera_image_to_bgr(image)
            if args.mask_output_size == "camera":
                mask_output_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
            else:
                mask_output_size = (int(seg_input_size[0]), int(seg_input_size[1]))
            front_stats = split_camera.process(
                int(image.frame),
                frame_bgr,
                mask_output_size=mask_output_size,
            )

            gt_3class: Optional[np.ndarray] = None
            if gt_queue is not None:
                gt_image = od_oai.wait_for_camera_frame(
                    gt_queue,
                    world_frame,
                    args.camera_timeout,
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

            result = result_store.wait_for(int(image.frame), args.result_timeout)
            remote_stats = None
            mask: Optional[np.ndarray] = None
            if result is not None:
                remote_stats = {
                    "server_ms": float(result["server_ms"]),
                    "round_trip_ms": (time.perf_counter() - float(result["camera_sent_perf"])) * 1000.0,
                    "mask_payload_bytes_estimate": int(result.get("mask_payload_bytes_estimate", 0)),
                }
                candidate = result.get("mask")
                if isinstance(candidate, np.ndarray):
                    mask = candidate

            if metrics_logger is not None:
                if metrics_warmup_remaining > 0:
                    metrics_warmup_remaining -= 1
                    if metrics_warmup_remaining == 0:
                        metrics_start_perf = time.perf_counter()
                else:
                    if metrics_start_perf is None:
                        metrics_start_perf = time.perf_counter()
                    metrics_logger.append(
                        build_metrics_record(
                            frame_id=int(image.frame),
                            elapsed_s=time.perf_counter() - metrics_start_perf,
                            args=args,
                            front_stats=front_stats,
                            remote_stats=remote_stats,
                            mask=mask,
                            gt_3class=gt_3class,
                        )
                    )

            if gui_enabled:
                annotated = draw_segmentation_overlay(
                    frame_bgr,
                    mask,
                    front_stats,
                    remote_stats,
                    args=args,
                    metrics_warmup_remaining=metrics_warmup_remaining,
                )
                cv2.imshow("CARLA Split Segmentation OAI", annotated)
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
        for sock in (camera_sender, camera_receiver, remote_receiver, remote_sender):
            if sock is None:
                continue
            try:
                sock.close()
            except OSError:
                pass
        if remote_worker is not None:
            remote_worker.join(timeout=1.0)
        result_receiver.join(timeout=1.0)
        if gui_enabled:
            cv2.destroyAllWindows()
        if metrics_logger is not None:
            metrics_logger.close()
            print(f"Saved metrics CSV to {metrics_csv_path}")


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")

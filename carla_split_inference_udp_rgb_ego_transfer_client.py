#!/usr/bin/env python3

"""RGB-only ego-view transferability client for SceneSense OD/SEG.

This experiment harness owns one CARLA ego vehicle and one RGB camera, then
feeds each frame into the existing RGB-only split-inference task paths:

* OD: Faster R-CNN front half -> UDP -> Faster R-CNN back half.
* SEG: LR-ASPP front half -> UDP -> LR-ASPP back half.

It supports moving/autopilot ego and static parked-ego viewpoints so we can
measure model transferability without changing the clean single-ego controller
architecture script. Use --seg-route pole_trained to evaluate the pole-trained
RGB-only LR-ASPP checkpoint on the parked ego.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import carla_split_inference_udp_data_collect as od_collect
import carla_split_inference_udp_oai as od_oai
import carla_split_inference_udp_segmentation_demo as seg_demo
import carla_split_inference_udp_segmentation_trained_lraspp_demo as trained_seg_demo
from scenesense_tx_gate import TxGateDecision, decision_to_stats


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = ROOT / "metrics_logs" / "rgb_ego_transfer"
DEFAULT_OD_PORT_BASE = 37100
DEFAULT_SEG_PORT_BASE = 37200
DEFAULT_POLE_LRASPP_EXPERIMENT = (
    ROOT
    / "experiments"
    / "pole_lraspp_training"
    / "20260505_173329_pole_lraspp_training"
)

carla = od_collect.carla
cv2 = od_collect.cv2


class TimerTaskGate:
    """In-memory OD/SEG timer gate with the same decide() shape as TxGate."""

    def __init__(
        self,
        *,
        od_seconds: float,
        seg_seconds: float,
        startup_task: str,
        profile: str,
        log_csv: Optional[Path],
    ) -> None:
        self.od_seconds = max(0.001, float(od_seconds))
        self.seg_seconds = max(0.001, float(seg_seconds))
        self.startup_task = str(startup_task).strip().lower()
        self.profile = str(profile or "")
        self.started_perf = time.perf_counter()
        self._lock = threading.Lock()
        self._last_logged_task: Optional[str] = None
        self._log_file = None
        self._writer: Optional[csv.DictWriter] = None

        if log_csv is not None:
            log_csv.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_csv.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._log_file,
                fieldnames=("wall_time_iso", "elapsed_s", "active_task", "profile"),
            )
            self._writer.writeheader()
            self._log_file.flush()

    def reset(self) -> None:
        with self._lock:
            self.started_perf = time.perf_counter()
            self._last_logged_task = None

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def current_task(self) -> str:
        elapsed = time.perf_counter() - self.started_perf
        cycle = self.od_seconds + self.seg_seconds
        phase = elapsed % cycle
        if self.startup_task == "seg":
            return "seg" if phase < self.seg_seconds else "od"
        return "od" if phase < self.od_seconds else "seg"

    def _log_if_changed(self, active_task: str) -> None:
        with self._lock:
            if active_task == self._last_logged_task:
                return
            self._last_logged_task = active_task
            if self._writer is None:
                return
            self._writer.writerow(
                {
                    "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "elapsed_s": time.perf_counter() - self.started_perf,
                    "active_task": active_task,
                    "profile": self.profile,
                }
            )
            if self._log_file is not None:
                self._log_file.flush()

    def decide(self, task_name: str) -> TxGateDecision:
        normalized_task = str(task_name or "").strip().lower()
        active_task = self.current_task()
        self._log_if_changed(active_task)
        active = normalized_task == active_task
        return TxGateDecision(
            active=active,
            task_name=normalized_task,
            active_task=active_task,
            control_file="in-memory-timer",
            updated_at=time.time(),
            reason="match" if active else "muted_by_controller",
            profile=self.profile,
        )


class TaskGateView:
    """Task-specific adapter passed into the existing front-half classes."""

    def __init__(self, gate: TimerTaskGate, task_name: str) -> None:
        self.gate = gate
        self.task_name = task_name

    def decide(self) -> TxGateDecision:
        return self.gate.decide(self.task_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RGB-only OD/SEG split-inference transferability checks from "
            "moving/autopilot or static parked ego viewpoints."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--vehicle-blueprint", default="vehicle.lincoln.mkz_2017")
    parser.add_argument(
        "--ego-mode",
        choices=("autopilot", "parked"),
        default="autopilot",
        help="Use a moving/autopilot ego or a frozen parked ego sensor platform.",
    )
    parser.add_argument("--ego-spawn-index", type=int, default=-1)
    parser.add_argument(
        "--ego-freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vehicle physics in --ego-mode parked.",
    )
    parser.add_argument("--ego-spawn-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--ego-spawn-right-offset-m", type=float, default=0.0)
    parser.add_argument("--ego-spawn-z-offset-m", type=float, default=0.15)
    parser.add_argument("--ego-spawn-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--ego-spawn-anchor-x",
        type=float,
        default=None,
        help=(
            "When --ego-mode parked and --ego-spawn-index is unset, prefer the "
            "vehicle spawn point nearest this world x coordinate."
        ),
    )
    parser.add_argument(
        "--ego-spawn-anchor-y",
        type=float,
        default=None,
        help=(
            "When --ego-mode parked and --ego-spawn-index is unset, prefer the "
            "vehicle spawn point nearest this world y coordinate."
        ),
    )
    parser.add_argument(
        "--ego-spawn-anchor-label",
        default="",
        help="Optional human-readable label for the parked-spawn anchor.",
    )
    parser.add_argument(
        "--camera-resolution",
        choices=["custom", *od_collect.CAMERA_RESOLUTION_PRESETS.keys()],
        default="custom",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=384)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.6)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.7)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--camera-warmup-ticks", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=30)
    parser.add_argument("--weather-preset", default=od_collect.WEATHER_PRESET_NONE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--front-device", default="auto")
    parser.add_argument("--back-device", default="auto")

    parser.add_argument("--run-duration-s", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--metrics-warmup-frames", type=int, default=0)
    parser.add_argument("--metrics-log-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--run-tag-prefix", default="rgb_ego_transfer")
    parser.add_argument("--profile", default="rgb_ego_transfer")
    parser.add_argument("--od-seconds", type=float, default=10.0)
    parser.add_argument("--seg-seconds", type=float, default=5.0)
    parser.add_argument("--startup-task", choices=("od", "seg"), default="od")
    parser.add_argument(
        "--compute-muted-fronts",
        action="store_true",
        help=(
            "Also run the inactive task front half. Default is off so the muted "
            "task is quiet in both compute and network payload."
        ),
    )

    parser.add_argument("--od-port-base", type=int, default=DEFAULT_OD_PORT_BASE)
    parser.add_argument("--seg-port-base", type=int, default=DEFAULT_SEG_PORT_BASE)
    parser.add_argument("--chunk-bytes", type=int, default=60000)
    parser.add_argument("--socket-timeout", type=float, default=0.25)
    parser.add_argument("--result-timeout", type=float, default=0.35)

    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--max-detections", type=int, default=20)
    od_gt_group = parser.add_mutually_exclusive_group()
    od_gt_group.add_argument(
        "--enable-od-gt",
        dest="enable_od_gt",
        action="store_true",
        help=(
            "Project CARLA vehicle/person actors into the RGB camera and log "
            "OD precision/recall using class-aware 2D IoU matching."
        ),
    )
    od_gt_group.add_argument(
        "--disable-od-gt",
        dest="enable_od_gt",
        action="store_false",
        help="Skip CARLA actor GT projection for OD quality metrics.",
    )
    parser.set_defaults(enable_od_gt=False)
    parser.add_argument("--od-gt-iou-threshold", type=float, default=0.5)
    parser.add_argument("--od-gt-min-area-px", type=float, default=64.0)
    parser.add_argument("--od-gt-max-distance-m", type=float, default=80.0)
    parser.add_argument("--rcnn-min-size", type=int, default=0)
    parser.add_argument("--rcnn-max-size", type=int, default=0)
    parser.add_argument("--bypass-rcnn-transform", action="store_true")

    parser.add_argument(
        "--segmentation-model",
        choices=seg_demo.SEGMENTATION_MODEL_CHOICES,
        default=seg_demo.SEGMENTATION_MODEL_LRASPP,
    )
    parser.add_argument(
        "--seg-route",
        choices=("moving", "generic", "pole_trained"),
        default="moving",
        help=(
            "moving/generic = original RGB-only torchvision LR-ASPP SEG route; "
            "pole_trained = RGB-only LR-ASPP checkpoint trained from pole views."
        ),
    )
    seg_pretrained = parser.add_mutually_exclusive_group()
    seg_pretrained.add_argument("--seg-pretrained", dest="seg_pretrained", action="store_true")
    seg_pretrained.add_argument(
        "--seg-disable-pretrained", dest="seg_pretrained", action="store_false"
    )
    parser.set_defaults(seg_pretrained=True)
    parser.add_argument("--seg-weights-path", default="")
    parser.add_argument(
        "--trained-experiment-dir",
        default=str(DEFAULT_POLE_LRASPP_EXPERIMENT),
        help="Pole LR-ASPP experiment directory used when --seg-route pole_trained.",
    )
    parser.add_argument("--seg-num-classes", type=int, default=21)
    parser.add_argument(
        "--seg-class-scheme",
        choices=("voc", "carla_3class"),
        default="voc",
    )
    parser.add_argument(
        "--use-checkpoint-input-size",
        dest="use_checkpoint_input_size",
        action="store_true",
    )
    parser.add_argument(
        "--disable-checkpoint-input-size",
        dest="use_checkpoint_input_size",
        action="store_false",
    )
    parser.set_defaults(use_checkpoint_input_size=True)
    parser.add_argument("--seg-input-width", type=int, default=512)
    parser.add_argument("--seg-input-height", type=int, default=288)
    parser.add_argument("--seg-mask-strength", type=float, default=0.72)
    parser.add_argument("--enable-semantic-gt", action="store_true")

    parser.add_argument(
        "--quantization-mode",
        choices=od_collect.QUANT_MODE_CHOICES,
        default=od_collect.QUANT_MODE_PER_TENSOR_UINT8,
    )
    parser.add_argument(
        "--entropy-coder",
        choices=od_collect.ENTROPY_CODER_CHOICES,
        default=od_collect.ENTROPY_CODER_ZLIB,
    )
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--roi-objectness-threshold", type=float, default=0.0)
    parser.add_argument(
        "--ae-mode",
        choices=od_collect.AE_MODE_CHOICES,
        default=od_collect.AE_MODE_OFF,
    )
    parser.add_argument("--ae-bottleneck-channels", type=int, default=64)
    parser.add_argument("--ae-spatial-stride", type=int, default=1)
    parser.add_argument("--ae-checkpoint", default="")
    parser.add_argument("--ae-seed", type=int, default=0)
    parser.add_argument("--per-level-compress-probe", action="store_true")

    for task in ("od", "seg"):
        parser.add_argument(
            f"--{task}-quantization-mode",
            choices=od_collect.QUANT_MODE_CHOICES,
            default=None,
        )
        parser.add_argument(
            f"--{task}-entropy-coder",
            choices=od_collect.ENTROPY_CODER_CHOICES,
            default=None,
        )
        parser.add_argument(f"--{task}-zstd-level", type=int, default=None)
        parser.add_argument(f"--{task}-roi-objectness-threshold", type=float, default=None)
        parser.add_argument(
            f"--{task}-ae-mode",
            choices=od_collect.AE_MODE_CHOICES,
            default=None,
        )
        parser.add_argument(f"--{task}-ae-bottleneck-channels", type=int, default=None)
        parser.add_argument(f"--{task}-ae-spatial-stride", type=int, default=None)
        parser.add_argument(f"--{task}-ae-checkpoint", default=None)

    return parser.parse_args()


def _task_args(args: argparse.Namespace, task: str) -> argparse.Namespace:
    data = dict(vars(args))
    data["run_tag"] = f"{args.run_tag_prefix}_{task}"
    data["metrics_log_prefix"] = f"{args.run_tag_prefix}_{task}"
    data["tx_task_name"] = task
    data["tx_gate_file"] = ""
    data["tx_gate_default_inactive"] = True
    data["tx_gate_stale_timeout_s"] = 0.0
    data["collect_metrics"] = True
    data["live_plot"] = False
    data["drive_mode"] = "autopilot"
    data["manifest_extra_json"] = ""
    if task == "seg" and str(data.get("seg_route", "moving")) == "pole_trained":
        data["seg_pretrained"] = False
        data["seg_num_classes"] = 3
        data["seg_class_scheme"] = "carla_3class"
    for key in (
        "quantization_mode",
        "entropy_coder",
        "zstd_level",
        "roi_objectness_threshold",
        "ae_mode",
        "ae_bottleneck_channels",
        "ae_spatial_stride",
        "ae_checkpoint",
    ):
        override = data.get(f"{task}_{key}")
        if override is not None:
            data[key] = override
    return argparse.Namespace(**data)


def _port_tuple(base: int) -> Tuple[int, int, int, int]:
    return int(base), int(base) + 1, int(base) + 2, int(base) + 3


def _sanitize(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._-") or "run"


def _resolve_pole_trained_checkpoint(exp_dir: Path) -> Path:
    """Find the best pole-trained LR-ASPP checkpoint in a copied experiment."""
    exp_dir = exp_dir.expanduser().resolve()
    manifest_path = exp_dir / "manifest.json"
    for source_path in (manifest_path,):
        if not source_path.exists():
            continue
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        best = payload.get("best_checkpoint") if isinstance(payload, dict) else None
        if not best:
            continue
        candidate = Path(str(best)).expanduser()
        if not candidate.is_absolute():
            candidate = (exp_dir / candidate).resolve()
        if candidate.exists():
            return candidate

    best_path: Optional[Path] = None
    best_miou = -float("inf")
    for summary_path in sorted((exp_dir / "checkpoints").glob("*/trial_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        try:
            miou = float(summary.get("best_miou", float("nan")))
        except (TypeError, ValueError):
            continue
        candidates = []
        raw_best = str(summary.get("best_checkpoint", "") or "")
        if raw_best:
            candidates.append(Path(raw_best).expanduser())
        candidates.append(summary_path.parent / "best.pt")
        for candidate in candidates:
            if not candidate.is_absolute():
                candidate = (exp_dir / candidate).resolve()
            if candidate.exists() and miou > best_miou:
                best_miou = miou
                best_path = candidate
                break
    if best_path is None:
        raise FileNotFoundError(f"No pole-trained best.pt found under {exp_dir}")
    return best_path


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
            x=float(transform.location.x)
            + forward_x * float(forward_m)
            + right_x * float(right_m),
            y=float(transform.location.y)
            + forward_y * float(forward_m)
            + right_y * float(right_m),
            z=float(transform.location.z) + float(z_offset_m),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw) + float(yaw_offset_deg),
            roll=float(transform.rotation.roll),
        ),
    )


def _spawn_parked_ego_vehicle(
    *,
    world: "carla.World",
    args: argparse.Namespace,
) -> "carla.Actor":
    preferred, fell_back = od_collect.resolve_hero_blueprint(
        world, str(args.vehicle_blueprint)
    )
    if fell_back:
        print(
            f"Requested ego blueprint {args.vehicle_blueprint!r} was not found. "
            f"Falling back to {preferred.id!r}."
        )

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No CARLA spawn points are available for parked ego.")
    if int(args.ego_spawn_index) >= 0:
        index = int(args.ego_spawn_index) % len(spawn_points)
        ordered = [spawn_points[index], *spawn_points[:index], *spawn_points[index + 1 :]]
    elif args.ego_spawn_anchor_x is not None and args.ego_spawn_anchor_y is not None:
        anchor_x = float(args.ego_spawn_anchor_x)
        anchor_y = float(args.ego_spawn_anchor_y)
        ordered = sorted(
            spawn_points,
            key=lambda point: math.hypot(
                float(point.location.x) - anchor_x,
                float(point.location.y) - anchor_y,
            ),
        )
    else:
        ordered = list(spawn_points)
        random.shuffle(ordered)

    for spawn_point in ordered:
        blueprint = od_collect.get_fresh_vehicle_blueprint(
            world,
            preferred.id,
            "scenesense_rgb_parked_ego",
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
            actor.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
            )
        except RuntimeError:
            pass
        if bool(args.ego_freeze):
            try:
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass
        return actor

    raise RuntimeError("Unable to spawn parked ego vehicle at any available spawn point.")


def _output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.metrics_log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _sanitize(str(args.run_tag_prefix))
    return {
        "od_csv": output_dir / f"{prefix}_od_{stamp}.csv",
        "seg_csv": output_dir / f"{prefix}_seg_{stamp}.csv",
        "gate_csv": output_dir / f"{prefix}_gate_events_{stamp}.csv",
        "manifest": output_dir / f"{prefix}_manifest_{stamp}.json",
    }


def _make_socket_group(
    *,
    base_port: int,
    chunk_bytes: int,
    socket_timeout: float,
    transport: od_collect.TransportConfig,
) -> Tuple[
    od_collect.UDPMessageSocket,
    od_collect.UDPMessageSocket,
    od_collect.UDPMessageSocket,
    od_collect.UDPMessageSocket,
]:
    camera_source, remote_port, remote_source, camera_result = _port_tuple(base_port)
    return (
        od_collect.UDPMessageSocket(
            bind_port=camera_source,
            remote_port=remote_port,
            chunk_bytes=chunk_bytes,
            socket_timeout=socket_timeout,
            entropy_coder=transport.make_entropy_coder(),
        ),
        od_collect.UDPMessageSocket(
            bind_port=remote_port,
            remote_port=None,
            chunk_bytes=chunk_bytes,
            socket_timeout=socket_timeout,
            entropy_coder=transport.make_entropy_coder(),
        ),
        od_collect.UDPMessageSocket(
            bind_port=remote_source,
            remote_port=camera_result,
            chunk_bytes=chunk_bytes,
            socket_timeout=socket_timeout,
            entropy_coder=transport.make_entropy_coder(),
        ),
        od_collect.UDPMessageSocket(
            bind_port=camera_result,
            remote_port=None,
            chunk_bytes=chunk_bytes,
            socket_timeout=socket_timeout,
            entropy_coder=transport.make_entropy_coder(),
        ),
    )


def _idle_front_stats(decision: TxGateDecision) -> Dict[str, object]:
    return {
        "front_ms": 0.0,
        "payload_bytes": 0,
        "payload_bytes_uncompressed": 0,
        "payload_chunks": 0,
        **decision_to_stats(decision),
        "per_level_uncompressed_bytes": {},
        "per_level_compressed_bytes": {},
        "roi_drop_fraction_total": 0.0,
        "roi_drop_fraction_per_level": {},
    }


def _remote_stats(result: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if result is None:
        return None
    return {
        "server_ms": float(result["server_ms"]),
        "round_trip_ms": (time.perf_counter() - float(result["camera_sent_perf"])) * 1000.0,
    }


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    *,
    seg_args: argparse.Namespace,
    outputs: Dict[str, Path],
    town_loaded: str,
    weather_applied: Optional[str],
    camera_width: int,
    camera_height: int,
    camera_resolution_label: str,
    seg_input_size: Tuple[int, int],
    rcnn_min_size: Tuple[int, ...],
    rcnn_max_size: int,
) -> None:
    od_ports = _port_tuple(args.od_port_base)
    seg_ports = _port_tuple(args.seg_port_base)
    manifest = {
        "wall_time_iso": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "run_tag_prefix": args.run_tag_prefix,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "town_requested": args.town,
        "town_loaded": town_loaded,
        "weather_preset_requested": args.weather_preset,
        "weather_preset_applied": weather_applied or od_collect.WEATHER_PRESET_NONE,
        "camera_width": int(camera_width),
        "camera_height": int(camera_height),
        "camera_resolution_label": camera_resolution_label,
        "camera_fov": float(args.camera_fov),
        "camera_mount": {
            "x": float(args.camera_x),
            "y": float(args.camera_y),
            "z": float(args.camera_z),
            "pitch": float(args.camera_pitch),
            "yaw": float(args.camera_yaw),
            "roll": float(args.camera_roll),
        },
        "fps": float(args.fps),
        "npc_vehicles": int(args.npc_vehicles),
        "npc_pedestrians": int(args.npc_pedestrians),
        "ego_mode": str(args.ego_mode),
        "ego_spawn": {
            "index": int(args.ego_spawn_index),
            "freeze": bool(args.ego_freeze),
            "forward_offset_m": float(args.ego_spawn_forward_offset_m),
            "right_offset_m": float(args.ego_spawn_right_offset_m),
            "z_offset_m": float(args.ego_spawn_z_offset_m),
            "yaw_offset_deg": float(args.ego_spawn_yaw_offset_deg),
            "anchor_x": (
                None if args.ego_spawn_anchor_x is None else float(args.ego_spawn_anchor_x)
            ),
            "anchor_y": (
                None if args.ego_spawn_anchor_y is None else float(args.ego_spawn_anchor_y)
            ),
            "anchor_label": str(args.ego_spawn_anchor_label or ""),
        },
        "timer_policy": {
            "startup_task": args.startup_task,
            "od_seconds": float(args.od_seconds),
            "seg_seconds": float(args.seg_seconds),
            "profile": args.profile,
            "compute_muted_fronts": bool(args.compute_muted_fronts),
        },
        "ports": {
            "od": {
                "camera_source": od_ports[0],
                "remote_receive": od_ports[1],
                "remote_source": od_ports[2],
                "camera_result": od_ports[3],
            },
            "seg": {
                "camera_source": seg_ports[0],
                "remote_receive": seg_ports[1],
                "remote_source": seg_ports[2],
                "camera_result": seg_ports[3],
            },
        },
        "od": {
            "weights_path": str(args.weights_path or ""),
            "disable_pretrained": bool(args.disable_pretrained),
            "score_threshold": float(args.score_threshold),
            "max_detections": int(args.max_detections),
            "enable_od_gt": bool(args.enable_od_gt),
            "od_gt_iou_threshold": float(args.od_gt_iou_threshold),
            "od_gt_min_area_px": float(args.od_gt_min_area_px),
            "od_gt_max_distance_m": float(args.od_gt_max_distance_m),
            "rcnn_min_size_resolved": list(rcnn_min_size),
            "rcnn_max_size_resolved": int(rcnn_max_size),
            "bypass_rcnn_transform": bool(args.bypass_rcnn_transform),
        },
        "seg": {
            "seg_route": str(args.seg_route),
            "segmentation_model": seg_args.segmentation_model,
            "seg_pretrained": bool(seg_args.seg_pretrained),
            "seg_weights_path": str(seg_args.seg_weights_path or ""),
            "trained_experiment_dir": str(seg_args.trained_experiment_dir or ""),
            "seg_num_classes": int(seg_args.seg_num_classes),
            "seg_class_scheme": str(seg_args.seg_class_scheme),
            "seg_input_width": int(seg_input_size[0]),
            "seg_input_height": int(seg_input_size[1]),
            "use_checkpoint_input_size": bool(seg_args.use_checkpoint_input_size),
            "enable_semantic_gt": bool(seg_args.enable_semantic_gt),
        },
        "transport_common_defaults": {
            "quantization_mode": args.quantization_mode,
            "entropy_coder": args.entropy_coder,
            "zstd_level": int(args.zstd_level),
            "roi_objectness_threshold": float(args.roi_objectness_threshold),
            "ae_mode": args.ae_mode,
            "ae_bottleneck_channels": int(args.ae_bottleneck_channels),
            "ae_spatial_stride": int(args.ae_spatial_stride),
            "ae_checkpoint": str(args.ae_checkpoint or ""),
            "per_level_compress_probe": bool(args.per_level_compress_probe),
            "chunk_bytes": int(args.chunk_bytes),
            "socket_timeout": float(args.socket_timeout),
            "result_timeout": float(args.result_timeout),
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if carla is None:
        raise RuntimeError(f"CARLA Python API is not available: {od_collect._CARLA_IMPORT_ERROR}")

    random.seed(7)
    front_device = od_collect.resolve_device(args.front_device)
    back_device = od_collect.resolve_device(args.back_device)
    camera_width, camera_height, camera_resolution_label = od_collect.resolve_camera_dimensions(args)
    od_args = _task_args(args, "od")
    seg_args = _task_args(args, "seg")
    seg_runtime = trained_seg_demo if str(args.seg_route) == "pole_trained" else seg_demo
    if str(args.seg_route) == "pole_trained":
        if not str(seg_args.seg_weights_path or "").strip():
            seg_args.seg_weights_path = str(
                _resolve_pole_trained_checkpoint(
                    Path(str(seg_args.trained_experiment_dir))
                )
            )
        trained_seg_demo.apply_trained_checkpoint_args(seg_args)
    seg_input_size = seg_runtime._resolve_seg_input_size(
        seg_args, camera_width, camera_height
    )
    outputs = _output_paths(args)

    od_args.camera_source_port, od_args.remote_port, od_args.remote_source_port, od_args.camera_result_port = _port_tuple(args.od_port_base)
    seg_args.camera_source_port, seg_args.remote_port, seg_args.remote_source_port, seg_args.camera_result_port = _port_tuple(args.seg_port_base)

    if front_device.type == "cuda" or back_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("Building OD split model...")
    od_front_raw = od_collect.build_detector_model(od_args)
    od_back_raw = od_collect.clone_detector_model(od_front_raw)
    rcnn_min_size, rcnn_max_size = od_collect.apply_rcnn_transform_overrides(
        od_front_raw, od_args
    )
    od_collect.apply_rcnn_transform_overrides(od_back_raw, od_args)
    od_transport = od_collect.transport_config_from_args(od_args)
    od_front_ae = od_collect.build_feature_autoencoder(od_args, front_device)
    od_back_ae = od_collect.build_feature_autoencoder(od_args, back_device)

    print("Building SEG split model...")
    seg_front_raw, seg_back_raw = seg_runtime.build_segmentation_models(seg_args)
    seg_front_model = seg_runtime.TorchvisionSegmentationSplitModel(
        seg_front_raw,
        front_device,
        input_size=seg_input_size,
    )
    seg_back_model = seg_runtime.TorchvisionSegmentationSplitModel(
        seg_back_raw,
        back_device,
        input_size=seg_input_size,
    )
    seg_transport = od_collect.TransportConfig(
        quantization_mode=str(seg_args.quantization_mode),
        entropy_coder_name=str(seg_args.entropy_coder),
        zstd_level=int(seg_args.zstd_level),
        roi_objectness_threshold=float(seg_args.roi_objectness_threshold),
        bypass_rcnn_transform=False,
    )
    seg_front_ae = seg_runtime.build_per_level_autoencoder(seg_args, front_device)
    seg_back_ae = seg_runtime.build_per_level_autoencoder(seg_args, back_device)

    gate = TimerTaskGate(
        od_seconds=float(args.od_seconds),
        seg_seconds=float(args.seg_seconds),
        startup_task=args.startup_task,
        profile=args.profile,
        log_csv=outputs["gate_csv"],
    )

    od_sockets = _make_socket_group(
        base_port=args.od_port_base,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        transport=od_transport,
    )
    seg_sockets = _make_socket_group(
        base_port=args.seg_port_base,
        chunk_bytes=args.chunk_bytes,
        socket_timeout=args.socket_timeout,
        transport=seg_transport,
    )

    stop_event = threading.Event()
    od_result_store = od_collect.DetectionResultStore()
    seg_result_store = seg_runtime.SegmentationResultStore()

    od_front = od_collect.CameraSideSplitInference(
        od_front_raw,
        od_sockets[0],
        front_device,
        transport=od_transport,
        autoencoder=od_front_ae,
        per_level_compress_probe=bool(od_args.per_level_compress_probe),
        tx_gate=TaskGateView(gate, "od"),
    )
    od_worker = od_collect.RemoteInferenceWorker(
        model=od_back_raw,
        receiver=od_sockets[1],
        sender=od_sockets[2],
        device=back_device,
        score_threshold=float(od_args.score_threshold),
        max_detections=int(od_args.max_detections),
        stop_event=stop_event,
        transport=od_transport,
        autoencoder=od_back_ae,
    )
    od_receiver = od_collect.CameraResultReceiver(
        receiver=od_sockets[3],
        result_store=od_result_store,
        stop_event=stop_event,
    )

    seg_front = seg_runtime.CameraSideSegmentationSplitInference(
        seg_front_model,
        seg_sockets[0],
        transport=seg_transport,
        autoencoder=seg_front_ae,
        per_level_compress_probe=bool(seg_args.per_level_compress_probe),
        tx_gate=TaskGateView(gate, "seg"),
    )
    seg_worker = seg_runtime.SegmentationRemoteInferenceWorker(
        model=seg_back_model,
        receiver=seg_sockets[1],
        sender=seg_sockets[2],
        device=back_device,
        stop_event=stop_event,
        transport=seg_transport,
        autoencoder=seg_back_ae,
    )
    seg_receiver = seg_runtime.CameraResultReceiver(
        receiver=seg_sockets[3],
        result_store=seg_result_store,
        stop_event=stop_event,
    )

    od_worker.start()
    od_receiver.start()
    seg_worker.start()
    seg_receiver.start()

    od_logger = od_oai.MetricsCSVLogger(outputs["od_csv"])
    seg_logger = seg_runtime.MetricsCSVLogger(outputs["seg_csv"])

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    original_settings = world.get_settings()
    weather_applied = None
    actors: List["carla.Actor"] = []
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
    gt_queue: Optional["queue.Queue[carla.Image]"] = None
    gui_enabled = od_collect.has_graphical_display() and not bool(args.headless)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / max(0.001, float(args.fps))
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        world.tick()
        weather_applied = od_collect.apply_weather_preset(world, args.weather_preset)

        print(f"Connected to CARLA at {args.host}:{args.port}")
        print(f"Town: {world.get_map().name}")
        print(f"Camera: {camera_width}x{camera_height} ({camera_resolution_label}) @ {args.fps:g} FPS")
        print(f"Devices: front={front_device}, back={back_device}")
        print(f"Ego mode: {args.ego_mode}")
        print(f"SEG route: {args.seg_route}, checkpoint={seg_args.seg_weights_path or 'torchvision-default'}")
        print(
            "Timer policy: "
            f"{args.startup_task} first, OD={args.od_seconds:g}s, SEG={args.seg_seconds:g}s, "
            f"compute_muted_fronts={bool(args.compute_muted_fronts)}"
        )
        print(f"OD metrics: {outputs['od_csv']}")
        print(f"SEG metrics: {outputs['seg_csv']}")
        print(f"Gate events: {outputs['gate_csv']}")
        print(f"Manifest: {outputs['manifest']}")
        print(
            "OD ports: "
            f"{od_args.camera_source_port}->{od_args.remote_port}, "
            f"{od_args.remote_source_port}->{od_args.camera_result_port}"
        )
        print(
            "SEG ports: "
            f"{seg_args.camera_source_port}->{seg_args.remote_port}, "
            f"{seg_args.remote_source_port}->{seg_args.camera_result_port}"
        )

        if str(args.ego_mode) == "parked":
            hero_vehicle = _spawn_parked_ego_vehicle(world=world, args=args)
        else:
            hero_vehicle = od_collect.spawn_hero_vehicle(
                client,
                world,
                traffic_manager,
                args.vehicle_blueprint,
            )
        actors.append(hero_vehicle)
        print(f"Hero vehicle: {hero_vehicle.type_id}")
        try:
            ego_transform = hero_vehicle.get_transform()
            print(
                "Hero transform: "
                f"x={ego_transform.location.x:.2f}, y={ego_transform.location.y:.2f}, "
                f"z={ego_transform.location.z:.2f}, yaw={ego_transform.rotation.yaw:.1f}"
            )
        except RuntimeError:
            pass

        background_vehicles = od_collect.spawn_background_traffic(
            client,
            world,
            traffic_manager,
            int(args.npc_vehicles),
            hero_vehicle,
        )
        actors.extend(background_vehicles)
        if background_vehicles:
            print(f"Spawned {len(background_vehicles)} background vehicles.")

        walkers, controllers = od_collect.spawn_background_pedestrians(
            client,
            world,
            int(args.npc_pedestrians),
            hero_vehicle,
        )
        actors.extend(walkers)
        actors.extend(controllers)
        if walkers:
            print(f"Spawned {len(walkers)} background pedestrians.")

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(camera_width))
        camera_bp.set_attribute("image_size_y", str(camera_height))
        camera_bp.set_attribute("fov", str(float(args.camera_fov)))
        camera_bp.set_attribute("sensor_tick", str(1.0 / max(0.001, float(args.fps))))
        camera_transform = carla.Transform(
            carla.Location(
                x=float(args.camera_x),
                y=float(args.camera_y),
                z=float(args.camera_z),
            ),
            carla.Rotation(
                pitch=float(args.camera_pitch),
                yaw=float(args.camera_yaw),
                roll=float(args.camera_roll),
            ),
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=hero_vehicle)
        actors.append(camera)
        camera.listen(lambda image: od_collect.put_latest(image_queue, image))

        first_image = od_collect.warmup_camera_stream(
            world,
            image_queue,
            int(args.camera_warmup_ticks),
            float(args.camera_timeout),
        )
        print(f"Camera ready on frame {first_image.frame}.")

        gt_camera = None
        if bool(args.enable_semantic_gt):
            gt_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
            gt_bp.set_attribute("image_size_x", str(camera_width))
            gt_bp.set_attribute("image_size_y", str(camera_height))
            gt_bp.set_attribute("fov", str(float(args.camera_fov)))
            gt_bp.set_attribute("sensor_tick", str(1.0 / max(0.001, float(args.fps))))
            gt_camera = world.spawn_actor(gt_bp, camera_transform, attach_to=hero_vehicle)
            actors.append(gt_camera)
            gt_queue = queue.Queue(maxsize=2)
            gt_camera.listen(lambda image, q=gt_queue: od_collect.put_latest(q, image))
            print("Semantic GT camera enabled for SEG mIoU logging.")

        if gui_enabled:
            cv2.namedWindow("SceneSense Single-Ego OD/SEG Coordinator", cv2.WINDOW_AUTOSIZE)
        else:
            print("Headless run active. Press Ctrl+C to stop.")

        measurement_frames_logged = 0
        metrics_start_perf: Optional[float] = None
        warmup_remaining = max(0, int(args.metrics_warmup_frames))
        if warmup_remaining > 0:
            print(f"Metrics warm-up: skipping {warmup_remaining} frame(s).")
        gate_started = False

        while True:
            world_frame = int(world.tick())
            image = od_collect.wait_for_camera_frame(
                image_queue,
                world_frame,
                float(args.camera_timeout),
            )
            if image is None:
                print(
                    f"Warning: camera frame for world tick {world_frame} was not received.",
                    file=sys.stderr,
                )
                continue

            frame_bgr = od_collect.camera_image_to_bgr(image)
            frame_id = int(image.frame)
            if not gate_started:
                gate.reset()
                gate_started = True
            od_decision = gate.decide("od")
            seg_decision = gate.decide("seg")

            if bool(args.compute_muted_fronts) or od_decision.active:
                od_front_stats = od_front.process(frame_id, frame_bgr)
            else:
                od_front_stats = _idle_front_stats(od_decision)

            if bool(args.compute_muted_fronts) or seg_decision.active:
                seg_front_stats = seg_front.process(frame_id, frame_bgr)
            else:
                seg_front_stats = _idle_front_stats(seg_decision)

            od_result = (
                od_result_store.wait_for(frame_id, float(args.result_timeout))
                if int(od_front_stats.get("tx_active", 0))
                else None
            )
            seg_result = (
                seg_result_store.wait_for(frame_id, float(args.result_timeout))
                if int(seg_front_stats.get("tx_active", 0))
                else None
            )

            od_remote_stats = _remote_stats(od_result)
            seg_remote_stats = _remote_stats(seg_result)
            detections = list(od_result.get("detections", [])) if od_result is not None else []
            mask = (
                seg_result.get("mask")
                if seg_result is not None and isinstance(seg_result.get("mask"), np.ndarray)
                else None
            )

            gt_3class: Optional[np.ndarray] = None
            if gt_queue is not None:
                gt_image = od_collect.wait_for_camera_frame(
                    gt_queue,
                    world_frame,
                    float(args.camera_timeout),
                )
                if gt_image is not None:
                    gt_tags = seg_runtime.carla_semantic_image_to_tags(gt_image)
                    gt_3class = seg_runtime.map_carla_tags_to_3class(gt_tags)

            if warmup_remaining > 0:
                warmup_remaining -= 1
                if warmup_remaining == 0:
                    metrics_start_perf = time.perf_counter()
                continue

            if metrics_start_perf is None:
                metrics_start_perf = time.perf_counter()
            elapsed_s = time.perf_counter() - metrics_start_perf

            gt_objects: Optional[List[Dict[str, object]]] = None
            if bool(args.enable_od_gt):
                gt_objects = od_oai.project_od_ground_truth_objects(
                    world,
                    camera,
                    int(hero_vehicle.id),
                    width=int(camera_width),
                    height=int(camera_height),
                    fov=float(args.camera_fov),
                    max_distance_m=float(args.od_gt_max_distance_m),
                    min_area_px=float(args.od_gt_min_area_px),
                )
            od_logger.append(
                od_oai.build_metrics_record(
                    frame_id=frame_id,
                    elapsed_s=elapsed_s,
                    front_stats=od_front_stats,
                    remote_stats=od_remote_stats,
                    detections=detections,
                    gt_objects=gt_objects,
                    args=args,
                )
            )
            seg_logger.append(
                seg_runtime.build_metrics_record(
                    frame_id=frame_id,
                    elapsed_s=elapsed_s,
                    args=seg_args,
                    front_stats=seg_front_stats,
                    remote_stats=seg_remote_stats,
                    mask=mask,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    camera_resolution_label=camera_resolution_label,
                    seg_input_size=seg_input_size,
                    town=world.get_map().name,
                    weather_preset=weather_applied or od_collect.WEATHER_PRESET_NONE,
                    gt_3class=gt_3class,
                )
            )
            measurement_frames_logged += 1

            if measurement_frames_logged % 10 == 0:
                od_logger.flush()
                seg_logger.flush()

            if gui_enabled:
                active_task = gate.current_task()
                if active_task == "seg":
                    annotated = seg_runtime.draw_segmentation_overlay(
                        frame_bgr,
                        mask,
                        seg_front_stats,
                        seg_remote_stats,
                        args=seg_args,
                        metrics_warmup_remaining=0,
                    )
                else:
                    annotated = od_collect.draw_overlay(
                        frame_bgr,
                        detections,
                        od_front_stats,
                        od_remote_stats,
                        metrics_warmup_remaining=0,
                    )
                cv2.putText(
                    annotated,
                    f"active task: {active_task.upper()}",
                    (10, annotated.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SceneSense Single-Ego OD/SEG Coordinator", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if int(args.max_frames) > 0 and measurement_frames_logged >= int(args.max_frames):
                print(f"Reached --max-frames={int(args.max_frames)}; stopping run.")
                break
            if float(args.run_duration_s) > 0.0 and elapsed_s >= float(args.run_duration_s):
                print(f"Reached --run-duration-s={float(args.run_duration_s):.1f}; stopping run.")
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
        for socket_obj in (*od_sockets, *seg_sockets):
            socket_obj.close()
        od_worker.join(timeout=1.0)
        od_receiver.join(timeout=1.0)
        seg_worker.join(timeout=1.0)
        seg_receiver.join(timeout=1.0)
        od_logger.flush()
        seg_logger.flush()
        od_logger.close()
        seg_logger.close()
        gate.close()
        if gui_enabled:
            cv2.destroyAllWindows()
        try:
            _write_manifest(
                outputs["manifest"],
                args,
                seg_args=seg_args,
                outputs=outputs,
                town_loaded=world.get_map().name,
                weather_applied=weather_applied,
                camera_width=camera_width,
                camera_height=camera_height,
                camera_resolution_label=camera_resolution_label,
                seg_input_size=seg_input_size,
                rcnn_min_size=rcnn_min_size,
                rcnn_max_size=rcnn_max_size,
            )
        except Exception as exc:
            print(f"Warning: unable to write manifest: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")

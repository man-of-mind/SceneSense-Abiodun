from __future__ import annotations

import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


NEU_COLLAB_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
RGB_WORKFLOW_ROOT = NEU_COLLAB_ROOT / "pole_lraspp_training"
DEFAULT_CONFIG = WORKFLOW_ROOT / "configs" / "fusion_full_run.yaml"
DEFAULT_EXPERIMENT_ROOT = NEU_COLLAB_ROOT / "experiments" / "pole_lraspp_multimodal_fusion"
PROJECT_PYTHON = Path("/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3")
CARLA_BIN = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/CarlaUnreal.sh")

CLASS_BACKGROUND = 0
CLASS_VEHICLE = 1
CLASS_PERSON = 2
CLASS_NAMES = ("background", "vehicle", "person")
VEHICLE_TAGS = {14, 15, 16, 17, 18, 19}
PERSON_TAGS = {4, 12, 13, 24, 25}

MANIFEST_FIELDS = (
    "experiment_id",
    "sample_id",
    "split",
    "rgb_path",
    "mask_path",
    "instance_raw_path",
    "radar_tensor_path",
    "radar_points_path",
    "frame_id",
    "radar_frame_id",
    "timestamp",
    "radar_timestamp",
    "traffic_light_id",
    "traffic_light_opendrive_id",
    "map_name",
    "camera_x",
    "camera_y",
    "camera_z",
    "camera_pitch",
    "camera_yaw",
    "camera_roll",
    "camera_fov",
    "camera_width",
    "camera_height",
    "camera_fx",
    "camera_fy",
    "camera_cx",
    "camera_cy",
    "camera_matrix_json",
    "camera_inverse_matrix_json",
    "radar_matrix_json",
    "radar_inverse_matrix_json",
    "radar_to_camera_matrix_json",
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "anchor_pitch",
    "anchor_yaw",
    "anchor_roll",
    "radar_horizontal_fov",
    "radar_vertical_fov",
    "radar_range_m",
    "radar_points",
    "radar_stationary_points",
    "radar_parked_evidence_points",
    "traffic_density",
    "pedestrian_density",
    "scenario_id",
    "view_id",
    "vehicle_pixels",
    "person_pixels",
)

OBJECT_BOX_FIELDS = (
    "experiment_id",
    "sample_id",
    "frame_id",
    "timestamp",
    "traffic_light_id",
    "scenario_id",
    "view_id",
    "label",
    "gt_actor_id",
    "gt_source",
    "gt_actor_type_id",
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
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "object_sensor_x",
    "object_sensor_y",
    "object_sensor_z",
    "object_yaw_deg",
    "object_velocity_x_mps",
    "object_velocity_y_mps",
    "object_velocity_z_mps",
    "object_speed_mps",
    "stationary_age_s",
    "stationary_label",
    "parked_label",
    "radar_support_points",
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_json(path: Path) -> Dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_yaml_or_json(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text) or {}
            if not isinstance(payload, dict):
                raise ValueError(f"Config {path} must contain a mapping.")
            return payload
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"PyYAML is required to read YAML config {path}. "
                "Install pyyaml or pass a JSON config."
            ) from exc
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must contain a mapping.")
    return payload


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=True)
        return
    except Exception:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def merge_dict(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str]) -> Dict:
    config = _load_yaml_or_json(DEFAULT_CONFIG)
    if path:
        config = merge_dict(config, _load_yaml_or_json(Path(path).expanduser()))
    return config


def create_experiment_dir(config: Dict, explicit_dir: Optional[str] = None) -> Path:
    if explicit_dir:
        exp_dir = Path(explicit_dir).expanduser().resolve()
    else:
        name = str(config.get("experiment_name", "pole_lraspp_multimodal_fusion"))
        exp_dir = DEFAULT_EXPERIMENT_ROOT / f"{now_stamp()}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def append_manifest_rows(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def append_object_box_rows(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBJECT_BOX_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OBJECT_BOX_FIELDS})


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stable_split(sample_id: str, ratios: Dict, seed: int) -> str:
    import hashlib

    train_ratio = float(ratios.get("train", 0.72))
    val_ratio = float(ratios.get("val", 0.14))
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def instance_image_to_tags(raw_bgra: np.ndarray) -> np.ndarray:
    if raw_bgra.ndim != 3 or raw_bgra.shape[2] < 3:
        raise ValueError(f"Expected BGRA image, got shape={raw_bgra.shape}")
    return raw_bgra[:, :, 2].astype(np.uint8)


def carla_semantic_tags_to_training_mask(tags: np.ndarray) -> np.ndarray:
    mask = np.zeros(tags.shape, dtype=np.uint8)
    mask[np.isin(tags, list(VEHICLE_TAGS))] = CLASS_VEHICLE
    mask[np.isin(tags, list(PERSON_TAGS))] = CLASS_PERSON
    return mask


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def class_iou_from_confusion(confusion: np.ndarray) -> Tuple[float, List[float], float]:
    ious: List[float] = []
    for cls in range(confusion.shape[0]):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - tp)
        fn = float(confusion[cls, :].sum() - tp)
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    valid = [value for value in ious if not math.isnan(value)]
    miou = float(np.mean(valid)) if valid else float("nan")
    pixel_acc = float(np.trace(confusion) / max(1.0, confusion.sum()))
    return miou, ious, pixel_acc


def update_confusion(confusion: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int) -> None:
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid].astype(np.int64) * int(num_classes) + pred[valid].astype(np.int64)
    bincount = np.bincount(encoded, minlength=int(num_classes) * int(num_classes))
    confusion += bincount.reshape(int(num_classes), int(num_classes))


def subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    paths = [str(WORKFLOW_ROOT), str(RGB_WORKFLOW_ROOT), str(NEU_COLLAB_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    stop_file: Optional[Path] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] RUN {' '.join(command)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        while True:
            code = proc.poll()
            if code is not None:
                return int(code)
            if stop_file is not None and stop_file.exists():
                log_fh.write(f"[{datetime.now().isoformat(timespec='seconds')}] stop_requested; terminating stage\n")
                log_fh.flush()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    code = proc.poll()
                    if code is not None:
                        return int(code)
                    time.sleep(0.5)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return int(proc.wait())
            time.sleep(1.0)


def load_csv_grouped(path: Path, key: str) -> Dict[str, List[Dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped

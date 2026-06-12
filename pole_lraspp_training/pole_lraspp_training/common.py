from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

NEU_COLLAB_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKFLOW_ROOT / "configs" / "default_config.json"
DEFAULT_EXPERIMENT_ROOT = NEU_COLLAB_ROOT / "experiments" / "pole_lraspp_training"
PROJECT_PYTHON = Path("/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3")

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
    "frame_id",
    "timestamp",
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
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_json(path: Path) -> Dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


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
    config = load_json(DEFAULT_CONFIG)
    if path:
        config = merge_dict(config, load_json(Path(path)))
    return config


def create_experiment_dir(config: Dict, explicit_dir: Optional[str] = None) -> Path:
    if explicit_dir:
        exp_dir = Path(explicit_dir).expanduser().resolve()
    else:
        name = str(config.get("experiment_name", "pole_lraspp_training"))
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


def stable_split(sample_id: str, ratios: Dict, seed: int) -> str:
    train_ratio = float(ratios.get("train", 0.72))
    val_ratio = float(ratios.get("val", 0.14))
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


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


def carla_semantic_tags_to_training_mask(tags: np.ndarray) -> np.ndarray:
    mask = np.zeros(tags.shape, dtype=np.uint8)
    mask[np.isin(tags, list(VEHICLE_TAGS))] = CLASS_VEHICLE
    mask[np.isin(tags, list(PERSON_TAGS))] = CLASS_PERSON
    return mask


def instance_image_to_tags(raw_bgra: np.ndarray) -> np.ndarray:
    # CARLA raw camera buffers are BGRA. Instance segmentation keeps the
    # semantic tag in the red channel in the same convention used by the
    # semantic camera path in the existing segmentation demo.
    if raw_bgra.ndim != 3 or raw_bgra.shape[2] < 3:
        raise ValueError(f"Expected BGRA image, got shape={raw_bgra.shape}")
    return raw_bgra[:, :, 2].astype(np.uint8)


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


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
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
        )
        return proc.wait()


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
    encoded = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    bincount = np.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bincount.reshape(num_classes, num_classes)

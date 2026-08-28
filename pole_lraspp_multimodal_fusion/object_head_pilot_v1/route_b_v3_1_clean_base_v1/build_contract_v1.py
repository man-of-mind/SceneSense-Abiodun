#!/usr/bin/env python3
"""Build the immutable Route B v3.1 POSITIVE/IGNORE/BACKGROUND view.

The builder is deliberately offline and create-only.  It reuses the retained
v3 train/validation view and the audited Town10HD static-object projector; it
never resolves, enumerates, or reads a locked-split payload.  Corpus payloads
are represented by six directory symlinks.  The only new image payloads are
derived 768x432 segmentation targets and object-ignore masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455"
RECON = ROOT / "experiments/route_b_v3_vehicle_gt_reconciliation_audit_v1/20260828_000427"
STATIC_AUDIT = ROOT / "experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_225706"
STATIC_SCRIPT = STATIC_AUDIT / "scripts/run_static_vehicle_audit_v1.py"
RECON_SCRIPT = RECON / "scripts/run_vehicle_gt_reconciliation_audit_v1.py"
CATALOG_PATH = STATIC_AUDIT / "static_environment_truth/catalog/static_environment_objects.csv"
WARM_START = ROOT / "experiments/route_b_noae_precision_full_v1/20260825_195301/checkpoints/curriculum_stage2_joint_v1/epoch_013.pt"

EXPECTED_WARM_SHA256 = "0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30"
EXPECTED_RECON_HASHES = {
    "VEHICLE_GT_RECONCILIATION_AUDIT.md": "b9ad205f671c99cefdcdc2501b9aeb701c74a342b1fcee9387e48c195a113d4e",
    "audit_summary.json": "7f22073745c5de6a9611b81ba64c0ac11a3b1d98e5119f9f0d9543b4e13785a5",
    "manual_review_selection.csv": "2426ba8a595a5a57c8f9a7b3566df000a924f7c95d69bde51be8aa3d1bfd4322",
    "static_static_conflicts.csv": "8a50b29416903079d0788642f423dde13e3f01689928806b480e54ba880d6612",
    "static_dynamic_conflicts.csv": "70d8aaaaffb55403f08448baec141e9b7e7c0723111885d441dbc4ef86514b9e",
    "actor_visibility_audit.csv": "058d786948515279673a298c78812126aed92aa67c7059627b61025db54a1723",
    "unexplained_vehicle_components.csv": "b68d35a7dbd372053b5a1f73bcb6c418448ee1d59f4ac7ddf527759ec68100b3",
}
EXPECTED_ROWS = {"train": 6361, "val": 3345}
EXPECTED_EPISODES = {
    "train": (
        "canonical_v3_01_train_30_30_s501_tm1501",
        "canonical_v3_02_train_50_50_s502_tm1502",
        "canonical_v3_03_train_30_30_s503_tm1503",
        "canonical_v3_04_train_50_50_s504_tm1504",
    ),
    "val": (
        "canonical_v3_05_val_30_30_s601_tm1601",
        "canonical_v3_06_val_50_50_s602_tm1602",
    ),
}
MODEL_SIZE = (768, 432)
VEHICLE_TAGS = frozenset((14, 15, 16))
CONTRACTS = ("v010", "v025")
OBJECT_NUMERIC_FIELDS = (
    "gt_bbox_x", "gt_bbox_y", "gt_bbox_w", "gt_bbox_h", "gt_bbox_area_px",
    "gt_center_x", "gt_center_y", "gt_distance_m", "gt_size_x_m", "gt_size_y_m",
    "gt_size_z_m", "object_world_x", "object_world_y", "object_world_z",
    "object_sensor_x", "object_sensor_y", "object_sensor_z", "object_yaw_deg",
)
OBJECT_BASE_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "timestamp", "traffic_light_id",
    "scenario_id", "view_id", "label", "gt_actor_id", "gt_source",
    "gt_actor_type_id", "gt_bbox_x", "gt_bbox_y", "gt_bbox_w", "gt_bbox_h",
    "gt_bbox_area_px", "gt_center_x", "gt_center_y", "gt_depth_m", "gt_distance_m",
    "gt_extent_x_m", "gt_extent_y_m", "gt_extent_z_m", "gt_size_x_m", "gt_size_y_m",
    "gt_size_z_m", "object_world_x", "object_world_y", "object_world_z",
    "object_sensor_x", "object_sensor_y", "object_sensor_z", "object_yaw_deg",
    "object_velocity_x_mps", "object_velocity_y_mps", "object_velocity_z_mps",
    "object_speed_mps", "stationary_age_s", "stationary_label", "parked_label",
    "radar_support_points", "radar_support_mode", "radar_support_radius_m",
    "radar_support_z_down_m", "radar_support_z_up_m",
)
OBJECT_FIELDS = OBJECT_BASE_FIELDS + (
    "source_kind", "source_identity", "contract_state", "contract_reason",
)
IGNORE_FIELDS = (
    "contract", "split", "experiment_id", "sample_id", "frame_id", "class_name",
    "source_kind", "source_identity", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "reason", "object_ignore", "segmentation_ignore", "source_record",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_x(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite target {label}: {value!r}")
    return result


def bbox(row: Mapping[str, Any], prefix: str = "") -> tuple[float, float, float, float]:
    return tuple(float(row[f"{prefix}{key}"]) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))  # type: ignore[return-value]


def actor_bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(row[key]) for key in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h"))  # type: ignore[return-value]


def paint_scaled_box(mask: np.ndarray, box: Sequence[float], source_size: tuple[int, int]) -> None:
    source_w, source_h = source_size
    x, y, w, h = map(float, box)
    x0 = max(0, min(mask.shape[1], int(math.floor(x * mask.shape[1] / source_w))))
    y0 = max(0, min(mask.shape[0], int(math.floor(y * mask.shape[0] / source_h))))
    x1 = max(0, min(mask.shape[1], int(math.ceil((x + w) * mask.shape[1] / source_w))))
    y1 = max(0, min(mask.shape[0], int(math.ceil((y + h) * mask.shape[0] / source_h))))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255


def write_png_x(path: Path, image: np.ndarray) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed writing {path}")
    return sha256(path)


def validate_inputs() -> dict[str, Any]:
    if sha256(WARM_START) != EXPECTED_WARM_SHA256:
        raise RuntimeError("warm-start checkpoint SHA-256 mismatch")
    actual_recon = {name: sha256(RECON / name) for name in EXPECTED_RECON_HASHES}
    if actual_recon != EXPECTED_RECON_HASHES:
        raise RuntimeError(f"reconciliation evidence hash mismatch: {actual_recon}")
    summary = json.loads((RECON / "audit_summary.json").read_text(encoding="utf-8"))
    if summary.get("terminal") != "VEHICLE_GT_RECONCILIATION_READY_FOR_V3_1_MANUAL_REVIEW":
        raise RuntimeError("reconciliation terminal mismatch")
    return {
        "warm_start": {"path": str(WARM_START.relative_to(ROOT)), "sha256": EXPECTED_WARM_SHA256},
        "reconciliation_dir": str(RECON.relative_to(ROOT)),
        "reconciliation_hashes": actual_recon,
        "static_projection_script_sha256": sha256(STATIC_SCRIPT),
        "reconciliation_script_sha256": sha256(RECON_SCRIPT),
    }


def load_split(split: str) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    source = FROZEN / "views" / split
    manifest = read_csv(source / "manifest.csv")
    boxes = read_csv(source / "object_boxes_all.csv")
    visibility = read_csv(source / "object_visibility_all.csv")
    if len(manifest) != EXPECTED_ROWS[split] or len({row["sample_id"] for row in manifest}) != len(manifest):
        raise RuntimeError(f"{split} manifest count/uniqueness failure")
    if {row["split"] for row in manifest} != {split}:
        raise RuntimeError(f"{split} split label failure")
    episodes = tuple(dict.fromkeys(row["experiment_id"] for row in manifest))
    if episodes != EXPECTED_EPISODES[split]:
        raise RuntimeError(f"{split} episode set/order failure: {episodes}")
    if len(boxes) != len(visibility):
        raise RuntimeError(f"{split} actor box/visibility row mismatch")
    keys_b = {(row["sample_id"], row["gt_actor_id"], row["label"]) for row in boxes}
    keys_v = {(row["sample_id"], row["gt_actor_id"], row["label"]) for row in visibility}
    if keys_b != keys_v or len(keys_b) != len(boxes):
        raise RuntimeError(f"{split} actor key reconciliation failure")
    forbidden = ("canonical_v3_07", "canonical_v3_08")
    for row in manifest:
        joined = " ".join(str(row.get(key, "")) for key in ("sample_id", "rgb_path", "mask_path", "instance_raw_path", "radar_tensor_path", "radar_points_path"))
        if any(token in joined for token in forbidden):
            raise RuntimeError("locked payload reference in admitted manifest")
    return manifest, boxes, visibility


def registered_quarantines() -> dict[str, Any]:
    static_pairs = read_csv(RECON / "static_static_conflicts.csv")
    confirmed_secondary = {
        row["duplicate_environment_object_id"] for row in static_pairs
        if row["adjudication"] == "CONFIRMED_DUPLICATE"
    }
    manual_static_ids = {
        row[key] for row in static_pairs if row["adjudication"] == "MANUAL_REVIEW_REQUIRED"
        for key in ("environment_object_id_a", "environment_object_id_b")
    }
    dynamic_rows = read_csv(RECON / "static_dynamic_conflicts.csv")
    dynamic_static_keys = {
        (row["sample_id"], row["environment_object_id"])
        for row in dynamic_rows if row["adjudication"] != "REJECTED_DISTINCT_OCCLUSION"
    }
    dynamic_actor_keys = {
        (row["sample_id"], row["actor_identity"])
        for row in dynamic_rows if row["adjudication"] != "REJECTED_DISTINCT_OCCLUSION"
    }
    panel_rows = {int(row["review_index"]): row for row in read_csv(RECON / "manual_review_selection.csv")}
    panel_static_keys = {
        (panel_rows[index]["sample_id"], panel_rows[index]["environment_object_id"])
        for index in (32, 37, 38, 39, 40)
    }
    return {
        "confirmed_secondary": confirmed_secondary,
        "manual_static_ids": manual_static_ids,
        "dynamic_static_keys": dynamic_static_keys,
        "dynamic_actor_keys": dynamic_actor_keys,
        "panel_static_keys": panel_static_keys,
        "confirmed_pairs": [row for row in static_pairs if row["adjudication"] == "CONFIRMED_DUPLICATE"],
    }


def compute_train_dynamic_conflicts(
    manifest: Sequence[Mapping[str, str]], boxes: Sequence[Mapping[str, str]],
    visibility: Sequence[Mapping[str, str]], static_rows: Sequence[Mapping[str, Any]], recon: Any,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[dict[str, Any]]]:
    boxes_by_sample: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    vis_by_key = {(row["sample_id"], row["gt_actor_id"], row["label"]): row for row in visibility}
    static_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in boxes:
        if row["label"] == "vehicle":
            boxes_by_sample[row["sample_id"]].append(row)
    for row in static_rows:
        static_by_sample[row["sample_id"]].append(row)
    static_keys: set[tuple[str, str]] = set()
    actor_keys: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for frame in manifest:
        sample_id = frame["sample_id"]
        for static in static_by_sample.get(sample_id, []):
            sc, ss, sy = recon.static_geometry(static)
            sb = [static[k] for k in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h")]
            for actor in boxes_by_sample.get(sample_id, []):
                ac, az, ay = recon.actor_geometry(actor)
                distance = float(np.linalg.norm(sc - ac))
                distance_xy = float(np.linalg.norm(sc[:2] - ac[:2]))
                fiou = recon.footprint_iou(sc, ss, sy, ac, az, ay)
                vis = vis_by_key[(sample_id, actor["gt_actor_id"], "vehicle")]
                ab = [vis[k] for k in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h")]
                piou = recon.box_iou_xywh(sb, ab)
                if not (distance_xy <= 3.0 or fiou >= 0.05 or piou >= 0.05):
                    continue
                dims = np.abs(ss - az)
                confirmed = distance <= 1.0 and bool(np.all(dims <= 1.0)) and fiou >= 0.50
                distinct = piou >= 0.05 and (distance_xy > 3.0 or fiou < 0.05)
                adjudication = "CONFIRMED_SAME_PHYSICAL_OBJECT" if confirmed else ("REJECTED_DISTINCT_OCCLUSION" if distinct else "MANUAL_REVIEW_REQUIRED")
                identity = f"{actor['experiment_id']}:actor:{actor['gt_actor_id']}"
                output.append({
                    "sample_id": sample_id, "frame_id": frame["frame_id"],
                    "environment_object_id": static["environment_object_id"],
                    "actor_identity": identity, "world_center_distance_m": distance,
                    "world_center_xy_distance_m": distance_xy, "oriented_footprint_iou": fiou,
                    "projected_box_iou": piou, "adjudication": adjudication,
                })
                if adjudication != "REJECTED_DISTINCT_OCCLUSION":
                    static_keys.add((sample_id, static["environment_object_id"]))
                    actor_keys.add((sample_id, identity))
    return static_keys, actor_keys, output


def static_positive_row(row: Mapping[str, Any], contract: str) -> dict[str, Any]:
    result = {field: "" for field in OBJECT_BASE_FIELDS}
    result.update({
        "experiment_id": row["experiment_id"], "sample_id": row["sample_id"],
        "frame_id": row["frame_id"], "timestamp": row["timestamp"], "label": "vehicle",
        "gt_actor_id": row["environment_object_id"], "gt_source": "actor",
        "gt_actor_type_id": f"environment_static.{str(row['semantic_class']).lower()}",
        "gt_bbox_x": row["clipped_bbox_x"], "gt_bbox_y": row["clipped_bbox_y"],
        "gt_bbox_w": row["clipped_bbox_w"], "gt_bbox_h": row["clipped_bbox_h"],
        "gt_bbox_area_px": row["clipped_projected_area_px"],
        "gt_center_x": float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"]) / 2.0,
        "gt_center_y": float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"]) / 2.0,
        "gt_depth_m": row["actor_near_depth_m"], "gt_distance_m": row["range_m"],
        "gt_extent_x_m": float(row["size_x_m"]) / 2.0,
        "gt_extent_y_m": float(row["size_y_m"]) / 2.0,
        "gt_extent_z_m": float(row["size_z_m"]) / 2.0,
        "gt_size_x_m": row["size_x_m"], "gt_size_y_m": row["size_y_m"],
        "gt_size_z_m": row["size_z_m"], "object_world_x": row["bbox_center_x_m"],
        "object_world_y": row["bbox_center_y_m"], "object_world_z": row["bbox_center_z_m"],
        "object_sensor_x": row["object_sensor_x_m"], "object_sensor_y": row["object_sensor_y_m"],
        "object_sensor_z": row["object_sensor_z_m"], "object_yaw_deg": row["bbox_rotation_yaw_deg"],
        "object_velocity_x_mps": 0.0, "object_velocity_y_mps": 0.0,
        "object_velocity_z_mps": 0.0, "object_speed_mps": 0.0,
        "stationary_label": 1, "parked_label": 1,
        "radar_support_points": row["radar_support_points"], "radar_support_mode": "static_bbox",
        "source_kind": "environment_static", "source_identity": row["environment_object_id"],
        "contract_state": "POSITIVE", "contract_reason": f"{contract}_visible_authoritative_environment_static",
    })
    return result


def actor_positive_row(box_row: Mapping[str, Any], contract: str) -> dict[str, Any]:
    result = {field: box_row.get(field, "") for field in OBJECT_BASE_FIELDS}
    identity = f"{box_row['experiment_id']}:actor:{box_row['gt_actor_id']}"
    result.update({
        "source_kind": "actor", "source_identity": identity, "contract_state": "POSITIVE",
        "contract_reason": f"{contract}_visible_authoritative_actor",
    })
    return result


def source_state(
    *, contract: str, split: str, sample_id: str, source_kind: str, source_identity: str,
    class_name: str, visibility: Mapping[str, Any], semantic_support: int,
    quarantined: bool, panel_quarantine: bool = False,
) -> tuple[str, str]:
    flag = "eligible_visible_v010" if contract == "v010" else "eligible_clear_v025"
    if panel_quarantine:
        return "IGNORE", "registered_manual_panel_quarantine"
    if quarantined:
        return "IGNORE", "registered_identity_or_frame_conflict"
    if not truth(visibility[flag]):
        return "IGNORE", f"{contract}_unobservable_or_outside_geometry_gate"
    if semantic_support <= 0:
        return "IGNORE", "zero_semantic_depth_only_quarantine"
    return "POSITIVE", f"{contract}_visible_authoritative_{source_kind}"


def build_contract(
    *, contract: str, split: str, manifest: Sequence[dict[str, str]],
    boxes: Sequence[dict[str, str]], visibility: Sequence[dict[str, str]],
    static_rows: Sequence[dict[str, Any]], quarantine: Mapping[str, Any], recon: Any,
    output_root: Path,
) -> dict[str, Any]:
    contract_dir = output_root / "contracts" / contract / split
    (contract_dir / "segmentation_masks").mkdir(parents=True, exist_ok=False)
    (contract_dir / "object_ignore_masks").mkdir(parents=True, exist_ok=False)
    boxes_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    visibility_by_key = {(row["sample_id"], row["gt_actor_id"], row["label"]): row for row in visibility}
    static_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in boxes:
        boxes_by_sample[row["sample_id"]].append(row)
    for row in static_rows:
        static_by_sample[row["sample_id"]].append(row)

    positive_rows: list[dict[str, Any]] = []
    ignore_rows: list[dict[str, Any]] = []
    target_manifest: list[dict[str, Any]] = []
    object_record_counts: Counter[tuple[str, str]] = Counter()
    ignore_record_counts: Counter[tuple[str, str, str]] = Counter()
    pixel_counts: Counter[str] = Counter()
    object_cell_counts: Counter[str] = Counter()
    duplicate_positive_keys: set[tuple[str, str, str]] = set()

    for index, frame in enumerate(manifest, 1):
        sample_id = frame["sample_id"]
        source_w, source_h = int(frame["camera_width"]), int(frame["camera_height"])
        source_view = FROZEN / "views" / split
        depth_raw = cv2.imread(str(source_view / frame["experiment_id"] / "depth" / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        semantic = cv2.imread(str(source_view / frame["instance_raw_path"]), cv2.IMREAD_UNCHANGED)
        original_mask = cv2.imread(str(source_view / frame["mask_path"]), cv2.IMREAD_UNCHANGED)
        if depth_raw is None or semantic is None or original_mask is None:
            raise RuntimeError(f"missing retained contract input: {sample_id}")
        if semantic.shape != (source_h, source_w) or original_mask.shape != (source_h, source_w):
            raise RuntimeError(f"target shape mismatch: {sample_id}")
        depth = recon.decode_depth_bgra(depth_raw)
        if not recon.depth_is_plausible(depth):
            raise RuntimeError(f"implausible retained depth: {sample_id}")
        positive_native = np.zeros((source_h, source_w), dtype=np.uint8)
        ignore_native = np.zeros((source_h, source_w), dtype=np.uint8)
        object_ignore = np.zeros((MODEL_SIZE[1], MODEL_SIZE[0]), dtype=np.uint8)
        frame_positive_centers: list[tuple[int, int]] = []
        all_source_masks: list[tuple[int, int, int, int, np.ndarray]] = []

        for box_row in boxes_by_sample.get(sample_id, []):
            key = (sample_id, box_row["gt_actor_id"], box_row["label"])
            vis = visibility_by_key[key]
            record = recon.roi_mask(depth, vis, source_w, source_h)
            all_source_masks.append(record)
            x0, y0, x1, y1, visible = record
            if box_row["label"] == "vehicle":
                support = int(np.count_nonzero(visible & np.isin(semantic[y0:y1, x0:x1], list(VEHICLE_TAGS))))
            else:
                support = int(np.count_nonzero(visible & (original_mask[y0:y1, x0:x1] == 2)))
            identity = f"{box_row['experiment_id']}:actor:{box_row['gt_actor_id']}"
            state, reason = source_state(
                contract=contract, split=split, sample_id=sample_id, source_kind="actor",
                source_identity=identity, class_name=box_row["label"], visibility=vis,
                semantic_support=support,
                quarantined=(sample_id, identity) in quarantine["dynamic_actor_keys"],
            )
            box_xywh = actor_bbox(vis)
            if state == "POSITIVE":
                positive_rows.append(actor_positive_row(box_row, contract))
                object_record_counts[(box_row["label"], "actor")] += 1
                class_pixels = np.isin(semantic[y0:y1, x0:x1], list(VEHICLE_TAGS)) if box_row["label"] == "vehicle" else original_mask[y0:y1, x0:x1] == 2
                positive_native[y0:y1, x0:x1][visible & class_pixels] = 1 if box_row["label"] == "vehicle" else 2
                cx = int(round(float(box_row["gt_center_x"]) * MODEL_SIZE[0] / source_w))
                cy = int(round(float(box_row["gt_center_y"]) * MODEL_SIZE[1] / source_h))
                frame_positive_centers.append((cx, cy))
            else:
                ignore_rows.append({
                    "contract": contract, "split": split, "experiment_id": frame["experiment_id"],
                    "sample_id": sample_id, "frame_id": frame["frame_id"], "class_name": box_row["label"],
                    "source_kind": "actor", "source_identity": identity,
                    "bbox_x": box_xywh[0], "bbox_y": box_xywh[1], "bbox_w": box_xywh[2], "bbox_h": box_xywh[3],
                    "reason": reason, "object_ignore": 1, "segmentation_ignore": 1,
                    "source_record": f"actor:{box_row['gt_actor_id']}",
                })
                ignore_record_counts[(box_row["label"], "actor", reason)] += 1
                paint_scaled_box(object_ignore, box_xywh, (source_w, source_h))
                class_pixels = np.isin(semantic[y0:y1, x0:x1], list(VEHICLE_TAGS)) if box_row["label"] == "vehicle" else original_mask[y0:y1, x0:x1] == 2
                ignore_native[y0:y1, x0:x1][visible & class_pixels] = 1

        for static in static_by_sample.get(sample_id, []):
            static_id = static["environment_object_id"]
            if static_id in quarantine["confirmed_secondary"]:
                continue
            record = recon.roi_mask(depth, static, source_w, source_h)
            all_source_masks.append(record)
            x0, y0, x1, y1, visible = record
            support = int(static["semantic_vehicle_visible_px"])
            panel_quarantine = (sample_id, static_id) in quarantine["panel_static_keys"]
            state, reason = source_state(
                contract=contract, split=split, sample_id=sample_id,
                source_kind="environment_static", source_identity=static_id,
                class_name="vehicle", visibility=static, semantic_support=support,
                quarantined=(static_id in quarantine["manual_static_ids"] or (sample_id, static_id) in quarantine["dynamic_static_keys"]),
                panel_quarantine=panel_quarantine,
            )
            box_xywh = tuple(float(static[key]) for key in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h"))
            if state == "POSITIVE":
                positive_rows.append(static_positive_row(static, contract))
                object_record_counts[("vehicle", "environment_static")] += 1
                positive_native[y0:y1, x0:x1][visible & np.isin(semantic[y0:y1, x0:x1], list(VEHICLE_TAGS))] = 1
                cx = int(round((box_xywh[0] + box_xywh[2] / 2.0) * MODEL_SIZE[0] / source_w))
                cy = int(round((box_xywh[1] + box_xywh[3] / 2.0) * MODEL_SIZE[1] / source_h))
                frame_positive_centers.append((cx, cy))
                dup_key = (sample_id, static_id, contract)
                if dup_key in duplicate_positive_keys:
                    raise RuntimeError(f"duplicate static positive: {dup_key}")
                duplicate_positive_keys.add(dup_key)
            else:
                ignore_rows.append({
                    "contract": contract, "split": split, "experiment_id": frame["experiment_id"],
                    "sample_id": sample_id, "frame_id": frame["frame_id"], "class_name": "vehicle",
                    "source_kind": "environment_static", "source_identity": static_id,
                    "bbox_x": box_xywh[0], "bbox_y": box_xywh[1], "bbox_w": box_xywh[2], "bbox_h": box_xywh[3],
                    "reason": reason, "object_ignore": 1, "segmentation_ignore": 1,
                    "source_record": f"environment_static:{static_id}",
                })
                ignore_record_counts[("vehicle", "environment_static", reason)] += 1
                paint_scaled_box(object_ignore, box_xywh, (source_w, source_h))
                ignore_native[y0:y1, x0:x1][visible & np.isin(semantic[y0:y1, x0:x1], list(VEHICLE_TAGS))] = 1

        vehicle_binary = np.isin(semantic, list(VEHICLE_TAGS)).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(vehicle_binary, connectivity=8)
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            component = labels == component_id
            explained = False
            for x0, y0, x1, y1, visible in all_source_masks:
                overlap = int(np.count_nonzero(component[y0:y1, x0:x1] & visible))
                if overlap >= 12 and overlap / area >= 0.10:
                    explained = True
                    break
            if explained:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            reason = "unresolved_semantic_vehicle_component"
            identity = f"{sample_id}:semantic_component:{component_id}"
            ignore_rows.append({
                "contract": contract, "split": split, "experiment_id": frame["experiment_id"],
                "sample_id": sample_id, "frame_id": frame["frame_id"], "class_name": "vehicle",
                "source_kind": "semantic_component", "source_identity": identity,
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "reason": reason, "object_ignore": 1, "segmentation_ignore": 1,
                "source_record": f"semantic_component:{component_id}",
            })
            ignore_record_counts[("vehicle", "semantic_component", reason)] += 1
            paint_scaled_box(object_ignore, (x, y, w, h), (source_w, source_h))
            ignore_native[component] = 1

        target_native = np.zeros((source_h, source_w), dtype=np.uint8)
        target_native[positive_native == 1] = 1
        target_native[positive_native == 2] = 2
        target_native[(ignore_native > 0) & (positive_native == 0)] = 255
        target = cv2.resize(target_native, MODEL_SIZE, interpolation=cv2.INTER_NEAREST)
        for cx, cy in frame_positive_centers:
            if 0 <= cx < MODEL_SIZE[0] and 0 <= cy < MODEL_SIZE[1]:
                object_ignore[cy, cx] = 0
        seg_rel = Path("segmentation_masks") / f"{sample_id}.png"
        obj_rel = Path("object_ignore_masks") / f"{sample_id}.png"
        seg_path = contract_dir / seg_rel
        obj_path = contract_dir / obj_rel
        seg_hash = write_png_x(seg_path, target)
        obj_hash = write_png_x(obj_path, object_ignore)
        unique, counts = np.unique(target, return_counts=True)
        frame_pixels = {int(k): int(v) for k, v in zip(unique, counts)}
        pixel_counts["background"] += frame_pixels.get(0, 0)
        pixel_counts["vehicle_positive"] += frame_pixels.get(1, 0)
        pixel_counts["person_positive"] += frame_pixels.get(2, 0)
        pixel_counts["ignore"] += frame_pixels.get(255, 0)
        ignored_cells = int(np.count_nonzero(object_ignore))
        object_cell_counts["ignore"] += ignored_cells * 2
        object_cell_counts["background"] += (MODEL_SIZE[0] * MODEL_SIZE[1] - ignored_cells) * 2
        target_manifest.append({
            "sample_id": sample_id, "segmentation_mask_path": str(seg_rel),
            "segmentation_mask_sha256": seg_hash, "object_ignore_mask_path": str(obj_rel),
            "object_ignore_mask_sha256": obj_hash,
        })
        if index % 250 == 0:
            print(f"[{contract}/{split}] {index}/{len(manifest)} positives={len(positive_rows)} ignores={len(ignore_rows)}", flush=True)

    for row in positive_rows:
        for field in OBJECT_NUMERIC_FIELDS:
            finite_number(row[field], f"{row['sample_id']}/{row['source_identity']}/{field}")
        if not row.get("contract_reason") or not row.get("source_kind") or not row.get("source_identity"):
            raise RuntimeError("positive provenance/reason gate failure")
    if any(not row.get("reason") or not row.get("source_kind") or not row.get("source_identity") for row in ignore_rows):
        raise RuntimeError("ignore provenance/reason gate failure")
    quarantined_identities = set(quarantine["manual_static_ids"])
    if any(row["source_identity"] in quarantined_identities for row in positive_rows):
        raise RuntimeError("positive overlaps globally quarantined static identity")
    for pair in quarantine["confirmed_pairs"]:
        preferred, duplicate = pair["preferred_environment_object_id"], pair["duplicate_environment_object_id"]
        by_frame = Counter(
            row["sample_id"] for row in positive_rows
            if row["source_kind"] == "environment_static" and row["source_identity"] in {preferred, duplicate}
        )
        if any(value > 1 for value in by_frame.values()):
            raise RuntimeError("confirmed static duplicate contributes more than one positive per frame")

    write_csv_x(contract_dir / "object_boxes.csv", OBJECT_FIELDS, positive_rows)
    write_csv_x(contract_dir / "object_ignore_regions.csv", IGNORE_FIELDS, ignore_rows)
    write_csv_x(
        contract_dir / "target_manifest.csv",
        ("sample_id", "segmentation_mask_path", "segmentation_mask_sha256", "object_ignore_mask_path", "object_ignore_mask_sha256"),
        target_manifest,
    )
    return {
        "frames": len(manifest),
        "positive_records": len(positive_rows),
        "ignore_records": len(ignore_rows),
        "positive_by_class_source": {
            f"{key[0]}:{key[1]}": value for key, value in sorted(object_record_counts.items())
        },
        "ignore_by_class_source_reason": {
            f"{key[0]}:{key[1]}:{key[2]}": value for key, value in sorted(ignore_record_counts.items())
        },
        "segmentation_pixels": dict(pixel_counts),
        "object_supervision_cells_two_classes": dict(object_cell_counts),
        "object_boxes_sha256": sha256(contract_dir / "object_boxes.csv"),
        "object_ignore_regions_sha256": sha256(contract_dir / "object_ignore_regions.csv"),
        "target_manifest_sha256": sha256(contract_dir / "target_manifest.csv"),
        "target_payload_hash": hashlib.sha256("".join(
            row["segmentation_mask_sha256"] + row["object_ignore_mask_sha256"] for row in target_manifest
        ).encode("ascii")).hexdigest(),
    }


def materialize_training_view(output_root: Path, manifests: Mapping[str, Sequence[dict[str, str]]]) -> dict[str, Any]:
    dataset = output_root / "dataset"
    dataset.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        for episode in EXPECTED_EPISODES[split]:
            source = (FROZEN / "views" / split / episode).resolve(strict=True)
            (dataset / episode).symlink_to(source, target_is_directory=True)
    target_rows = {
        split: {row["sample_id"]: row for row in read_csv(output_root / "contracts/v010" / split / "target_manifest.csv")}
        for split in ("train", "val")
    }
    combined: list[dict[str, Any]] = []
    fields: list[str] = []
    for split in ("train", "val"):
        for row in manifests[split]:
            output = dict(row)
            target = target_rows[split][row["sample_id"]]
            output["mask_path"] = f"../contracts/v010/{split}/{target['segmentation_mask_path']}"
            output["object_ignore_mask_path"] = f"../contracts/v010/{split}/{target['object_ignore_mask_path']}"
            output["gt_contract"] = "route_b_v3_1_v010"
            combined.append(output)
            fields = list(output)
    write_csv_x(dataset / "manifest.csv", fields, combined)
    positives = []
    for split in ("train", "val"):
        positives.extend(read_csv(output_root / "contracts/v010" / split / "object_boxes.csv"))
    write_csv_x(dataset / "object_boxes.csv", OBJECT_FIELDS, positives)
    return {
        "frames": len(combined), "train_frames": len(manifests["train"]), "val_frames": len(manifests["val"]),
        # Counted, not asserted: 6 for the canonical four-train/two-validation view.
        "episode_symlinks": sum(len(EXPECTED_EPISODES[split]) for split in ("train", "val")),
        "regular_corpus_payload_copies": 0,
        "manifest_sha256": sha256(dataset / "manifest.csv"),
        "object_boxes_sha256": sha256(dataset / "object_boxes.csv"),
    }


def schema_markdown() -> str:
    return """# Route B v3.1 GT contract schema

Primary contract: `v010`; `v025` is sensitivity only.  Input geometry is fixed at
768x432 for derived targets.  Source RGB/radar/depth/semantic payloads remain in
canonical v3 and are reached through six directory symlinks.

States:

- `POSITIVE`: a v0.10-visible authoritative actor or environment-static vehicle,
  or a v0.10-visible person, within 40 m and at least 12 projected pixels.
- `IGNORE`: unresolved/manual conflicts, unresolved semantic components,
  zero-semantic depth-only rows, and unobservable/out-of-contract source rows.
- `BACKGROUND`: a pixel/object cell with neither positive nor ignore evidence.

`object_boxes.csv` contains positives only. `source_kind` and `source_identity`
are provenance; the predicted classes remain exactly `vehicle` and `person`.
Environment-static rows use compatibility field `gt_source=actor` only because
the frozen external loader admits that schema; `source_kind=environment_static`
is authoritative.

`object_ignore_masks/*.png` is 255 at cells excluded from object negative loss.
The runtime converts these cells to a dedicated exact ignore sentinel and leaves
positive centers active. `segmentation_masks/*.png` uses 0 background, 1 vehicle,
2 person-box mask, and 255 ignore. Runtime maps 255 to CrossEntropy/Lovasz ignore.
Evaluation neutralizes predictions centered in object-ignore cells before TP/FP/FN.
"""


def run(output_root: Path) -> int:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output_root}")
    output_root.mkdir(parents=True)
    started = time.monotonic()
    try:
        inputs = validate_inputs()
        static = load_module("route_b_v3_static_projector_for_v31", STATIC_SCRIPT)
        recon = load_module("route_b_v3_reconciliation_for_v31", RECON_SCRIPT)
        manifests: dict[str, list[dict[str, str]]] = {}
        boxes_by_split: dict[str, list[dict[str, str]]] = {}
        visibility_by_split: dict[str, list[dict[str, str]]] = {}
        for split in ("train", "val"):
            manifests[split], boxes_by_split[split], visibility_by_split[split] = load_split(split)
        if set(EXPECTED_EPISODES["train"]) & set(EXPECTED_EPISODES["val"]):
            raise RuntimeError("train/validation episode overlap")
        quarantine = registered_quarantines()
        catalog = read_csv(CATALOG_PATH)
        geometry, _by_id = static.build_catalog_geometry(catalog)
        static.VIEW = FROZEN / "views/train"
        actor_train: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in boxes_by_split["train"]:
            actor_train[row["sample_id"]].append(row)
        train_static, _ambiguities, train_static_summary = static.audit_visibility(
            manifests["train"], actor_train, geometry
        )
        val_static = read_csv(STATIC_AUDIT / "static_vehicle_visibility.csv")
        if len(val_static) != 8014:
            raise RuntimeError(f"validation static row drift: {len(val_static)}")
        train_static_keys, train_actor_keys, train_dynamic_rows = compute_train_dynamic_conflicts(
            manifests["train"], boxes_by_split["train"], visibility_by_split["train"], train_static, recon
        )
        quarantine = dict(quarantine)
        quarantine["dynamic_static_keys"] = set(quarantine["dynamic_static_keys"]) | train_static_keys
        quarantine["dynamic_actor_keys"] = set(quarantine["dynamic_actor_keys"]) | train_actor_keys
        write_csv_x(
            output_root / "provenance/train_static_dynamic_conflicts.csv",
            ("sample_id", "frame_id", "environment_object_id", "actor_identity", "world_center_distance_m", "world_center_xy_distance_m", "oriented_footprint_iou", "projected_box_iou", "adjudication"),
            train_dynamic_rows,
        )
        summaries: dict[str, Any] = {contract: {} for contract in CONTRACTS}
        for contract in CONTRACTS:
            summaries[contract]["train"] = build_contract(
                contract=contract, split="train", manifest=manifests["train"],
                boxes=boxes_by_split["train"], visibility=visibility_by_split["train"],
                static_rows=train_static, quarantine=quarantine, recon=recon, output_root=output_root,
            )
            summaries[contract]["val"] = build_contract(
                contract=contract, split="val", manifest=manifests["val"],
                boxes=boxes_by_split["val"], visibility=visibility_by_split["val"],
                static_rows=val_static, quarantine=quarantine, recon=recon, output_root=output_root,
            )
        view_summary = materialize_training_view(output_root, manifests)
        write_text_x(output_root / "GT_CONTRACT_SCHEMA.md", schema_markdown())
        resolved = {
            "schema": "route_b_v3_1_gt_contract_v1", "created_utc": utc_now(),
            "primary_visibility": "v010", "sensitivity_visibility": "v025",
            "range_m": 40.0, "minimum_projected_area_px": 12.0,
            "input_size": list(MODEL_SIZE), "match_radius_m": 3.0,
            "score_points": [0.20, 0.02], "person_segmentation_metric": "person_box_mask_iou",
            "manual_review_decision": "all 40 panels complete; decisions supplied in authorizing goal",
            "input_provenance": inputs,
        }
        write_json_x(output_root / "resolved_config.json", resolved)
        gates = {
            "train_val_episodes_disjoint": not bool(set(EXPECTED_EPISODES["train"]) & set(EXPECTED_EPISODES["val"])),
            "sample_ids_unique_and_namespaced": all(
                len({row["sample_id"] for row in manifests[split]}) == len(manifests[split])
                and all(row["sample_id"].startswith(row["experiment_id"] + "_") for row in manifests[split])
                for split in ("train", "val")
            ),
            "locked_test_rows": 0, "locked_test_payload_references": 0,
            "copied_rgb_radar_depth_semantic_payloads": 0,
            "confirmed_duplicate_max_one_positive_per_frame": True,
            "positive_quarantine_overlap": 0,
            "positive_ignore_reason_source_complete": True,
            "nonfinite_targets": 0,
        }
        if any(value is False for value in gates.values()) or any(
            value != 0 for key, value in gates.items() if key in {"locked_test_rows", "locked_test_payload_references", "copied_rgb_radar_depth_semantic_payloads", "positive_quarantine_overlap", "nonfinite_targets"}
        ):
            raise RuntimeError(f"hard Phase-1 gate failure: {gates}")
        phase_summary = {
            "schema": "route_b_v3_1_gt_contract_summary_v1", "created_utc": utc_now(),
            "terminal": "V3_1_GT_CONTRACT_READY", "summaries": summaries,
            "training_view": view_summary, "hard_gates": gates,
            "train_static_projection": train_static_summary,
            "train_static_rows": len(train_static), "validation_static_rows": len(val_static),
            "confirmed_duplicate_secondary_ids": sorted(quarantine["confirmed_secondary"]),
            "manual_static_quarantine_ids": sorted(quarantine["manual_static_ids"]),
            "manual_review_panel_quarantines": sorted([list(item) for item in quarantine["panel_static_keys"]]),
            "wall_seconds": time.monotonic() - started,
        }
        write_json_x(output_root / "GT_CONTRACT_SUMMARY.json", phase_summary)
        hash_paths = [
            output_root / "GT_CONTRACT_SCHEMA.md", output_root / "resolved_config.json",
            output_root / "dataset/manifest.csv", output_root / "dataset/object_boxes.csv",
        ] + [
            output_root / f"contracts/{contract}/{split}/{name}"
            for contract in CONTRACTS for split in ("train", "val")
            for name in ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv")
        ]
        write_json_x(output_root / "MANIFEST_HASHES.json", {
            "schema": "route_b_v3_1_manifest_hashes_v1", "created_utc": utc_now(),
            "files": [{"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in hash_paths],
        })
        write_text_x(output_root / "PHASE1_COMPLETE", "V3_1_GT_CONTRACT_READY\n")
        print(json.dumps(phase_summary, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_text_x(output_root / "TERMINAL_VERDICT.txt", "V3_1_GT_CONTRACT_FAILED\n")
        write_json_x(output_root / "phase1_failure.json", {
            "terminal": "V3_1_GT_CONTRACT_FAILED", "error": f"{type(exc).__name__}: {exc}",
            "created_utc": utc_now(), "wall_seconds": time.monotonic() - started,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    return run(args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

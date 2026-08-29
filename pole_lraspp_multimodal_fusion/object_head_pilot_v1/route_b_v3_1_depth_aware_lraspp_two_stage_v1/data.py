from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from common import read_csv

ROOT = Path(__file__).resolve().parents[3]
MODEL_WIDTH, MODEL_HEIGHT = 768, 432
STRIDE = 4
GRID_WIDTH, GRID_HEIGHT = 192, 108
CLASS_NAMES = ("vehicle", "person")


def reference_gaussian_radius(height: float, width: float, overlap: float = 0.7) -> float:
    height, width, overlap = float(height), float(width), float(overlap)
    b1 = height + width
    c1 = width * height * (1.0 - overlap) / (1.0 + overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4.0 * c1))) / 2.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 16.0 * c2))) / 2.0
    a3 = 4.0 * overlap
    b3 = -2.0 * overlap * (height + width)
    c3 = (overlap - 1.0) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / 2.0
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center_x: float, center_y: float, radius: int) -> None:
    diameter = 2 * int(radius) + 1
    sigma = diameter / 6.0
    axis = np.arange(diameter, dtype=np.float32) - radius
    gaussian = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2.0 * sigma * sigma))
    x, y = int(math.floor(center_x)), int(math.floor(center_y))
    left, right = min(x, radius), min(heatmap.shape[1] - x, radius + 1)
    top, bottom = min(y, radius), min(heatmap.shape[0] - y, radius + 1)
    if min(left, right, top, bottom) < 0:
        return
    target = heatmap[y - top:y + bottom, x - left:x + right]
    source = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(target, source, out=target)


def conservative_ignore(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (MODEL_HEIGHT, MODEL_WIDTH):
        raise ValueError(mask.shape)
    return mask.reshape(GRID_HEIGHT, STRIDE, GRID_WIDTH, STRIDE).all(axis=(1, 3))


def _as_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float(default)


def load_visible_anchors(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_csv(path):
        key = (row["sample_id"], row["source_identity"])
        if key in result:
            raise RuntimeError(f"duplicate visible target key {key}")
        result[key] = row
    return result


def load_objects(dataset_root: Path, contract: str = "v010") -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split in ("train", "val"):
        for row in read_csv(Path(dataset_root) / f"contracts/{contract}/{split}/object_boxes.csv"):
            result[row["sample_id"]].append(row)
    return result


def _owner_record(row: Mapping[str, Any], frame: Mapping[str, str],
                  visible: Mapping[str, Any] | None) -> dict[str, Any]:
    width, height = float(frame["camera_width"]), float(frame["camera_height"])
    sx, sy = MODEL_WIDTH / width, MODEL_HEIGHT / height
    depth = _as_float(row, "object_sensor_x")
    right, up = _as_float(row, "object_sensor_y"), _as_float(row, "object_sensor_z")
    fx, fy = float(frame["camera_fx"]) * sx, float(frame["camera_fy"]) * sy
    cx, cy = float(frame["camera_cx"]) * sx, float(frame["camera_cy"]) * sy
    physical_u = cx + fx * right / depth
    physical_v = cy - fy * up / depth
    if row["label"] == "person":
        if visible is None:
            raise RuntimeError(f"missing visible anchor {row['sample_id']} {row['source_identity']}")
        anchor_x, anchor_y = float(visible["anchor_model_x"]), float(visible["anchor_model_y"])
        extent_w = float(visible["visible_bbox_grid_w"])
        extent_h = float(visible["visible_bbox_grid_h"])
        area = float(visible["visible_pixels"])
    else:
        anchor_x, anchor_y = _as_float(row, "gt_center_x") * sx, _as_float(row, "gt_center_y") * sy
        extent_w = _as_float(row, "gt_bbox_w") * sx / STRIDE
        extent_h = _as_float(row, "gt_bbox_h") * sy / STRIDE
        area = _as_float(row, "gt_bbox_area_px")
    grid_x, grid_y = anchor_x / STRIDE, anchor_y / STRIDE
    cell_x, cell_y = int(math.floor(grid_x)), int(math.floor(grid_y))
    if not (0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT):
        raise RuntimeError(f"eligible anchor out of native grid: {row['sample_id']} {row['source_identity']}")
    box_x = _as_float(row, "gt_center_x") * sx / STRIDE
    box_y = _as_float(row, "gt_center_y") * sy / STRIDE
    yaw = math.radians(_as_float(row, "object_yaw_deg"))
    return {
        "source_identity": row["source_identity"], "class_name": row["label"],
        "cell_x": cell_x, "cell_y": cell_y, "grid_x": grid_x, "grid_y": grid_y,
        "area": area, "depth": depth,
        "subcell": (grid_x - cell_x, grid_y - cell_y),
        "box_center_delta": (box_x - grid_x, box_y - grid_y),
        "box_wh": (_as_float(row, "gt_bbox_w") * sx / STRIDE,
                    _as_float(row, "gt_bbox_h") * sy / STRIDE),
        "physical_ray_delta": (physical_u / STRIDE - grid_x, physical_v / STRIDE - grid_y),
        "dimensions": (max(0.01, _as_float(row, "gt_size_x_m")),
                       max(0.01, _as_float(row, "gt_size_y_m")),
                       max(0.01, _as_float(row, "gt_size_z_m"))),
        "yaw": (math.sin(yaw), math.cos(yaw)),
        "parked": _as_float(row, "parked_label"),
        "radar_support": float(_as_float(row, "radar_support_points") > 0.0),
        "radius": int(max(1, round(reference_gaussian_radius(extent_h, extent_w, 0.7)))),
        "local_xyz": (depth, right, up),
        "intrinsic": (fx, fy, cx, cy),
    }


OWNER_FLOAT_FIELDS = (
    "subcell", "box_center_delta", "box_wh", "physical_ray_delta", "depth",
    "dimensions", "yaw", "parked", "radar_support", "local_xyz", "intrinsic",
)


def build_sparse_targets(rows: Sequence[Mapping[str, Any]], frame: Mapping[str, str],
                         visible_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
                         ignore_cells: np.ndarray) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    heatmap = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_NAMES}
    for row in rows:
        if row.get("contract_state") != "POSITIVE":
            continue
        depth = _as_float(row, "object_sensor_x")
        if not math.isfinite(depth) or not (0.0 < depth <= 40.0):
            raise RuntimeError(f"actor depth outside registered range: {depth}")
        visible = visible_lookup.get((row["sample_id"], row["source_identity"]))
        item = _owner_record(row, frame, visible)
        class_index = CLASS_NAMES.index(item["class_name"])
        draw_gaussian(heatmap[class_index], item["grid_x"], item["grid_y"], item["radius"])
        heatmap[class_index, item["cell_y"], item["cell_x"]] = 1.0
        candidates[item["class_name"]].append(item)
    owners: dict[str, dict[str, torch.Tensor]] = {}
    collisions: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in candidates[class_name]:
            by_cell[(item["cell_y"], item["cell_x"])].append(item)
        selected: list[dict[str, Any]] = []
        for cell, items in by_cell.items():
            ranked = sorted(items, key=lambda value: (-value["area"], value["depth"], value["source_identity"]))
            selected.append(ranked[0])
            for loser in ranked[1:]:
                collisions.append({
                    "sample_id": frame["sample_id"], "class_name": class_name,
                    "cell_y": cell[0], "cell_x": cell[1],
                    "owner": ranked[0]["source_identity"], "nonowner": loser["source_identity"],
                    "state": "SAME_CLASS_NATIVE_CELL_UNREPRESENTABLE",
                })
        selected.sort(key=lambda item: (item["cell_y"], item["cell_x"], item["source_identity"]))
        count = len(selected)
        def tensor(name: str, width: int) -> torch.Tensor:
            if count == 0:
                return torch.empty((0, width), dtype=torch.float32)
            values = []
            for item in selected:
                value = item[name]
                values.append([float(value)] if width == 1 else [float(x) for x in value])
            return torch.tensor(values, dtype=torch.float32)
        owners[class_name] = {
            "cells": (torch.tensor([[item["cell_y"], item["cell_x"]] for item in selected], dtype=torch.long)
                      if count else torch.empty((0, 2), dtype=torch.long)),
            "subcell": tensor("subcell", 2),
            "box_center_delta": tensor("box_center_delta", 2),
            "box_wh": tensor("box_wh", 2),
            "physical_ray_delta": tensor("physical_ray_delta", 2),
            "depth": tensor("depth", 1),
            "dimensions": tensor("dimensions", 3),
            "yaw": tensor("yaw", 2),
            "parked": tensor("parked", 1),
            "radar_support": tensor("radar_support", 1),
            "local_xyz": tensor("local_xyz", 3),
            "intrinsic": tensor("intrinsic", 4),
        }
        ignored = np.broadcast_to(ignore_cells, heatmap[class_index].shape)
        heatmap[class_index][(heatmap[class_index] == 0.0) & ignored] = -1.0
    return torch.from_numpy(heatmap), owners, collisions


class DepthCache:
    def __init__(self, cache_dir: Path, expected_rows: Sequence[Mapping[str, str]]) -> None:
        self.cache_dir = Path(cache_dir)
        index = read_csv(self.cache_dir / "index.csv")
        self.lookup = {row["sample_id"]: row for row in index}
        expected = {row["sample_id"] for row in expected_rows}
        if set(self.lookup) != expected:
            raise RuntimeError(f"depth cache keys {len(self.lookup)} != expected {len(expected)}")
        n = len(index)
        self.depth = np.memmap(self.cache_dir / "depth_forward_f16.bin", mode="r", dtype=np.float16,
                               shape=(n, GRID_HEIGHT, GRID_WIDTH))
        self.valid = np.memmap(self.cache_dir / "valid_u8.bin", mode="r", dtype=np.uint8,
                               shape=(n, GRID_HEIGHT, GRID_WIDTH))
        self.radar = np.memmap(self.cache_dir / "radar_consistency_f32.bin", mode="r", dtype=np.float32)

    def get(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.lookup[sample_id]
        index = int(row["row_index"])
        offset, count = int(row["radar_float_offset"]), int(row["radar_point_count"])
        points = np.asarray(self.radar[offset:offset + count * 3]).reshape(count, 3).copy()
        return (
            torch.from_numpy(np.asarray(self.depth[index], dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(self.valid[index], dtype=np.uint8).copy()).bool(),
            torch.from_numpy(points),
        )


class TrainingDataset(Dataset):
    """Training-only labels. Inference has a separate signature with no depth argument."""

    def __init__(self, dataset_root: Path, rows: Sequence[dict[str, str]],
                 object_rows: Mapping[str, Sequence[dict[str, str]]],
                 visible_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
                 depth_cache: DepthCache, seed: int) -> None:
        self.root = Path(dataset_root)
        self.dataset = self.root / "dataset"
        self.rows = list(rows)
        self.object_rows = object_rows
        self.visible_lookup = visible_lookup
        self.depth_cache = depth_cache
        self.seed = int(seed)
        self.epoch = 1
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _input(self, row: Mapping[str, str], augment: bool) -> tuple[torch.Tensor, tuple[int, int]]:
        image = Image.open(self.dataset / row["rgb_path"]).convert("RGB")
        original = image.size
        if augment:
            digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{row['sample_id']}".encode()).digest()
            rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
            if rng.rand() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
            if rng.rand() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.9, 1.1)))
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        payload = np.load(self.dataset / row["radar_tensor_path"])
        try:
            radar = np.asarray(payload["radar"] if isinstance(payload, np.lib.npyio.NpzFile) else payload,
                               dtype=np.float32)
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if radar.shape != (4, MODEL_HEIGHT, MODEL_WIDTH):
            raise RuntimeError(f"radar tensor contract drift {radar.shape}")
        return torch.cat([rgb, torch.from_numpy(np.ascontiguousarray(radar))], dim=0), original

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        fused, _ = self._input(row, True)
        mask_image = Image.open(self.dataset / row["mask_path"]).convert("L")
        mask_image = mask_image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.NEAREST)
        segmentation = torch.from_numpy(np.asarray(mask_image, dtype=np.int64).copy())
        segmentation[segmentation == 255] = -100
        ignore_image = Image.open(self.dataset / row["object_ignore_mask_path"]).convert("L")
        ignore_image = ignore_image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.NEAREST)
        ignore = conservative_ignore(np.asarray(ignore_image, dtype=np.uint8) != 0)
        heatmap, owners, collisions = build_sparse_targets(
            self.object_rows.get(row["sample_id"], ()), row, self.visible_lookup, ignore,
        )
        depth, depth_valid, radar_points = self.depth_cache.get(row["sample_id"])
        return {
            "input": fused, "segmentation": segmentation, "heatmap": heatmap,
            "owners": owners, "collisions": collisions, "dense_depth": depth,
            "dense_valid": depth_valid, "radar_points": radar_points,
            "sample_id": row["sample_id"],
        }


class InferenceDataset(Dataset):
    """Deployable dataset: deliberately has no depth-label or depth-cache argument."""

    def __init__(self, dataset_root: Path, rows: Sequence[dict[str, str]]) -> None:
        self.root = Path(dataset_root)
        self.dataset = self.root / "dataset"
        self.rows = list(rows)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        row = self.rows[index]
        image = Image.open(self.dataset / row["rgb_path"]).convert("RGB")
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        payload = np.load(self.dataset / row["radar_tensor_path"])
        try:
            radar = np.asarray(payload["radar"] if isinstance(payload, np.lib.npyio.NpzFile) else payload,
                               dtype=np.float32)
        finally:
            if hasattr(payload, "close"):
                payload.close()
        fused = torch.cat([rgb, torch.from_numpy(np.ascontiguousarray(radar))], dim=0)
        return fused, dict(row)


def collate_training(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input": torch.stack([item["input"] for item in items]),
        "segmentation": torch.stack([item["segmentation"] for item in items]),
        "heatmap": torch.stack([item["heatmap"] for item in items]),
        "dense_depth": torch.stack([item["dense_depth"] for item in items]),
        "dense_valid": torch.stack([item["dense_valid"] for item in items]),
        "radar_points": [item["radar_points"] for item in items],
        "sample_id": [item["sample_id"] for item in items],
        "collisions": [value for item in items for value in item["collisions"]],
        "owners": {},
    }
    for class_name in CLASS_NAMES:
        keys = items[0]["owners"][class_name].keys()
        merged: dict[str, torch.Tensor] = {}
        for key in keys:
            values = []
            for batch_index, item in enumerate(items):
                value = item["owners"][class_name][key]
                if key == "cells" and value.numel():
                    value = torch.cat([torch.full((value.shape[0], 1), batch_index, dtype=torch.long), value], dim=1)
                elif key == "cells":
                    value = torch.empty((0, 3), dtype=torch.long)
                values.append(value)
            merged[key] = torch.cat(values, dim=0)
        result["owners"][class_name] = merged
    return result


def collision_audit(rows: Sequence[dict[str, str]], object_rows: Mapping[str, Sequence[dict[str, str]]],
                    visible_lookup: Mapping[tuple[str, str], Mapping[str, Any]], dataset_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for frame in rows:
        ignore = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        _heatmap, owners, collisions = build_sparse_targets(
            object_rows.get(frame["sample_id"], ()), frame, visible_lookup, ignore,
        )
        for name in CLASS_NAMES:
            eligible[name] += sum(row["label"] == name for row in object_rows.get(frame["sample_id"], ()))
            counts[name] += sum(value["class_name"] == name for value in collisions)
            if owners[name]["cells"].shape[0] + sum(value["class_name"] == name for value in collisions) != sum(
                    row["label"] == name for row in object_rows.get(frame["sample_id"], ())):
                raise RuntimeError("collision denominator reconciliation failed")
        examples.extend(collisions[:max(0, 20 - len(examples))])
    return {
        "frames": len(rows), "eligible": dict(eligible), "same_class_collisions": dict(counts),
        "fractions": {name: counts[name] / eligible[name] if eligible[name] else 0.0 for name in CLASS_NAMES},
        "cross_class_overwrite": 0, "silent_truncation": 0, "examples": examples,
    }

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, Sampler

from common import read_csv

CONTENT_W, CONTENT_H = 768, 432
PAD_BOTTOM = 16
NETWORK_H = CONTENT_H + PAD_BOTTOM
CLASS_NAMES = ("vehicle", "person")
DEPTH_BINS = 32
DEPTH_MAX_M = 40.0


def depth_edges() -> torch.Tensor:
    return torch.expm1(torch.linspace(0.0, math.log1p(DEPTH_MAX_M), DEPTH_BINS + 1, dtype=torch.float32))


class DepthCache:
    """Read-only synchronized dense surface-depth/radar consistency cache."""

    def __init__(self, cache_dir: Path, rows: Sequence[Mapping[str, str]]) -> None:
        self.cache_dir = Path(cache_dir).resolve(strict=True)
        index = read_csv(self.cache_dir / "index.csv")
        self.lookup = {row["sample_id"]: row for row in index}
        expected = {row["sample_id"] for row in rows}
        if set(self.lookup) != expected or len(index) != len(rows):
            raise RuntimeError(f"depth-cache key drift: {len(index)} versus {len(rows)}")
        count = len(index)
        self.depth = np.memmap(self.cache_dir / "depth_forward_f16.bin", mode="r", dtype=np.float16,
                               shape=(count, CONTENT_H // 4, CONTENT_W // 4))
        self.valid = np.memmap(self.cache_dir / "valid_u8.bin", mode="r", dtype=np.uint8,
                               shape=(count, CONTENT_H // 4, CONTENT_W // 4))
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


def load_split_rows(dataset_root: Path, split: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv(Path(dataset_root) / "dataset/manifest.csv") if row["split"] == split]
    expected = 16827 if split == "train" else 3345
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise RuntimeError(f"{split} frame contract drift")
    return rows


def load_objects(dataset_root: Path, split: str, contract: str = "v010") -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(Path(dataset_root) / f"contracts/{contract}/{split}/object_boxes.csv"):
        if row["contract_state"] == "POSITIVE":
            result[row["sample_id"]].append(row)
    return result


def model_intrinsic(row: Mapping[str, str]) -> torch.Tensor:
    sx, sy = CONTENT_W / float(row["camera_width"]), CONTENT_H / float(row["camera_height"])
    return torch.tensor([
        [float(row["camera_fx"]) * sx, 0.0, float(row["camera_cx"]) * sx],
        [0.0, float(row["camera_fy"]) * sy, float(row["camera_cy"]) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def camera_extrinsic(row: Mapping[str, str]) -> torch.Tensor:
    return torch.tensor(json.loads(row["camera_matrix_json"]), dtype=torch.float64)


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float(default)


def target_from_rows(frame: Mapping[str, str], rows: Sequence[Mapping[str, str]],
                     segmentation: torch.Tensor, ignore: torch.Tensor) -> dict[str, Any]:
    source_w, source_h = float(frame["camera_width"]), float(frame["camera_height"])
    sx, sy = CONTENT_W / source_w, CONTENT_H / source_h
    intrinsic = model_intrinsic(frame)
    values: dict[str, list[Any]] = defaultdict(list)
    edges = depth_edges()
    for row in rows:
        class_name = row["label"]
        if class_name not in CLASS_NAMES:
            continue
        x0 = _float(row, "gt_bbox_x") * sx
        y0 = _float(row, "gt_bbox_y") * sy
        x1 = (_float(row, "gt_bbox_x") + _float(row, "gt_bbox_w")) * sx
        y1 = (_float(row, "gt_bbox_y") + _float(row, "gt_bbox_h")) * sy
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(CONTENT_W), x1), min(float(CONTENT_H), y1)
        if not (x1 > x0 and y1 > y0):
            raise RuntimeError(f"eligible degenerate box: {frame['sample_id']} {row['source_identity']}")
        depth = _float(row, "object_sensor_x")
        if not math.isfinite(depth) or not (0.0 < depth <= DEPTH_MAX_M):
            raise RuntimeError(f"eligible actor depth drift: {depth}")
        right, up = _float(row, "object_sensor_y"), _float(row, "object_sensor_z")
        physical_u = float(intrinsic[0, 2]) + float(intrinsic[0, 0]) * right / depth
        physical_v = float(intrinsic[1, 2]) - float(intrinsic[1, 1]) * up / depth
        depth_bin = int(torch.bucketize(torch.tensor(depth), edges[1:-1], right=False).item())
        lower, upper = float(edges[depth_bin]), float(edges[depth_bin + 1])
        z, zl, zu = math.log1p(depth), math.log1p(lower), math.log1p(upper)
        residual = (z - 0.5 * (zl + zu)) / max(1e-12, zu - zl)
        yaw = math.radians(_float(row, "object_yaw_deg"))
        values["boxes"].append([x0, y0, x1, y1])
        values["labels"].append(CLASS_NAMES.index(class_name))
        values["depth"].append(depth)
        values["depth_bin"].append(depth_bin)
        values["depth_residual"].append(residual)
        values["physical_uv"].append([physical_u, physical_v])
        values["local_xyz"].append([depth, right, up])
        values["dimensions"].append([
            max(0.01, _float(row, "gt_size_x_m")), max(0.01, _float(row, "gt_size_y_m")),
            max(0.01, _float(row, "gt_size_z_m")),
        ])
        values["yaw"].append([math.sin(yaw), math.cos(yaw)])
        values["source_identity"].append(row["source_identity"])
        values["radar_supported"].append(float(_float(row, "radar_support_points") > 0))
        values["distance"].append(_float(row, "gt_distance_m", depth))
        values["projected_size"].append(max(x1 - x0, y1 - y0))
    count = len(values["labels"])
    def floats(name: str, width: int | None = None) -> torch.Tensor:
        if count:
            tensor = torch.tensor(values[name], dtype=torch.float32)
            return tensor[:, None] if width == 1 and tensor.ndim == 1 else tensor
        return torch.empty((0,) if width is None else (0, width), dtype=torch.float32)
    return {
        "boxes": floats("boxes", 4),
        "labels": torch.tensor(values["labels"], dtype=torch.int64) if count else torch.empty(0, dtype=torch.int64),
        "depth": floats("depth", 1).squeeze(1),
        "depth_bin": torch.tensor(values["depth_bin"], dtype=torch.int64) if count else torch.empty(0, dtype=torch.int64),
        "depth_residual": floats("depth_residual", 1).squeeze(1),
        "physical_uv": floats("physical_uv", 2),
        "local_xyz": floats("local_xyz", 3),
        "dimensions": floats("dimensions", 3),
        "yaw": floats("yaw", 2),
        "radar_supported": floats("radar_supported", 1).squeeze(1),
        "distance": floats("distance", 1).squeeze(1),
        "projected_size": floats("projected_size", 1).squeeze(1),
        "source_identity": list(values["source_identity"]),
        "segmentation": segmentation,
        "ignore_mask": ignore,
        "intrinsic": intrinsic,
        "extrinsic": camera_extrinsic(frame),
        "sample_id": frame["sample_id"],
        "frame_id": frame["frame_id"],
        "original_size_wh": (CONTENT_W, CONTENT_H),
        "padding_ltrb": (0, 0, 0, PAD_BOTTOM),
    }


class RouteBDataset(Dataset):
    """Training view with synchronized targets; depth labels never enter inference."""

    def __init__(self, dataset_root: Path, split: str, seed: int,
                 depth_cache: DepthCache | None = None, augment: bool = False) -> None:
        self.root = Path(dataset_root).resolve(strict=True)
        self.dataset = self.root / "dataset"
        self.rows = load_split_rows(self.root, split)
        self.objects = load_objects(self.root, split, "v010")
        self.split, self.seed, self.epoch = split, int(seed), 1
        self.cache, self.augment = depth_cache, bool(augment)
        if self.augment and split != "train":
            raise ValueError("augmentation is train-only")
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _rgb(self, row: Mapping[str, str]) -> torch.Tensor:
        image = Image.open(self.dataset / row["rgb_path"]).convert("RGB")
        if self.augment:
            digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{row['sample_id']}".encode()).digest()
            rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
            if rng.rand() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
            if rng.rand() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.9, 1.1)))
        image = image.resize((CONTENT_W, CONTENT_H), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return (rgb - self.mean) / self.std

    def _radar(self, row: Mapping[str, str]) -> torch.Tensor:
        payload = np.load(self.dataset / row["radar_tensor_path"])
        try:
            radar = np.asarray(payload["radar"] if isinstance(payload, np.lib.npyio.NpzFile) else payload,
                               dtype=np.float32)
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if radar.shape != (4, CONTENT_H, CONTENT_W) or not np.isfinite(radar).all():
            raise RuntimeError(f"radar contract drift: {radar.shape}")
        return torch.from_numpy(np.ascontiguousarray(radar))

    def _mask(self, row: Mapping[str, str], key: str) -> torch.Tensor:
        image = Image.open(self.dataset / row[key]).convert("L")
        image = image.resize((CONTENT_W, CONTENT_H), Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(image, dtype=np.int64).copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        content = torch.cat([self._rgb(row), self._radar(row)], dim=0)
        fused = F.pad(content, (0, 0, 0, PAD_BOTTOM), value=0.0)
        segmentation = self._mask(row, "mask_path")
        segmentation[segmentation == 255] = -100
        ignore = self._mask(row, "object_ignore_mask_path").ne(0)
        target = target_from_rows(row, self.objects.get(row["sample_id"], ()), segmentation, ignore)
        result = {"input": fused, "target": target, "row": dict(row)}
        if self.cache is not None:
            depth, valid, radar_points = self.cache.get(row["sample_id"])
            result.update({"dense_depth": depth, "dense_valid": valid, "radar_points": radar_points})
        return result


class InferenceDataset(Dataset):
    """Deployable view with no target, semantic-label, object, or depth-cache path."""

    _ROW_FIELDS = (
        "sample_id", "frame_id", "rgb_path", "radar_tensor_path", "camera_width", "camera_height",
        "camera_fx", "camera_fy", "camera_cx", "camera_cy", "camera_matrix_json",
    )

    def __init__(self, dataset_root: Path, split: str = "val") -> None:
        self.root = Path(dataset_root).resolve(strict=True)
        self.dataset = self.root / "dataset"
        # Whitelist the deployable metadata so label-path fields from the canonical
        # manifest cannot propagate into edge inference records.
        self.rows = [{key: row[key] for key in self._ROW_FIELDS}
                     for row in load_split_rows(self.root, split)]
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def _rgb(self, row: Mapping[str, str]) -> torch.Tensor:
        image = Image.open(self.dataset / row["rgb_path"]).convert("RGB")
        image = image.resize((CONTENT_W, CONTENT_H), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return (rgb - self.mean) / self.std

    def _radar(self, row: Mapping[str, str]) -> torch.Tensor:
        payload = np.load(self.dataset / row["radar_tensor_path"])
        try:
            radar = np.asarray(payload["radar"] if isinstance(payload, np.lib.npyio.NpzFile) else payload,
                               dtype=np.float32)
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if radar.shape != (4, CONTENT_H, CONTENT_W) or not np.isfinite(radar).all():
            raise RuntimeError(f"radar contract drift: {radar.shape}")
        return torch.from_numpy(np.ascontiguousarray(radar))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str], dict[str, torch.Tensor]]:
        row = self.rows[index]
        content = torch.cat([self._rgb(row), self._radar(row)], dim=0)
        fused = F.pad(content, (0, 0, 0, PAD_BOTTOM), value=0.0)
        calibration = {"intrinsic": model_intrinsic(row), "extrinsic": camera_extrinsic(row)}
        return fused, dict(row), calibration


def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input": torch.stack([item["input"] for item in items]),
        "targets": [item["target"] for item in items],
        "rows": [item["row"] for item in items],
        "sample_ids": [item["target"]["sample_id"] for item in items],
    }
    if "dense_depth" in items[0]:
        result.update({
            "dense_depth": torch.stack([item["dense_depth"] for item in items]),
            "dense_valid": torch.stack([item["dense_valid"] for item in items]),
            "radar_points": [item["radar_points"] for item in items],
        })
    return result


class FrozenEpochSampler(Sampler[int]):
    """Deterministic canonical sample order, resumable by index."""

    def __init__(self, length: int, seed: int, epoch: int, start_index: int = 0) -> None:
        self.length, self.seed, self.epoch, self.start_index = int(length), int(seed), int(epoch), int(start_index)

    def order(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.length, generator=generator)

    def __iter__(self) -> Iterator[int]:
        yield from self.order()[self.start_index:].tolist()

    def __len__(self) -> int:
        return self.length - self.start_index

    def state_dict(self) -> dict[str, int]:
        return {"length": self.length, "seed": self.seed, "epoch": self.epoch, "start_index": self.start_index}


def training_priors(dataset_root: Path) -> dict[str, Any]:
    rows = read_csv(Path(dataset_root) / "contracts/v010/train/object_boxes.csv")
    edges = depth_edges().numpy()
    histogram = np.ones((2, DEPTH_BINS + 1), dtype=np.float64)  # finite Laplace prior, including overflow
    dimensions: dict[str, list[list[float]]] = {name: [] for name in CLASS_NAMES}
    counts = Counter()
    projected = {name: Counter() for name in CLASS_NAMES}
    for row in rows:
        if row["contract_state"] != "POSITIVE" or row["label"] not in CLASS_NAMES:
            continue
        class_index, class_name = CLASS_NAMES.index(row["label"]), row["label"]
        depth = _float(row, "object_sensor_x")
        bin_index = int(np.searchsorted(edges[1:-1], depth, side="left")) if depth <= DEPTH_MAX_M else DEPTH_BINS
        histogram[class_index, bin_index] += 1
        dimensions[class_name].append([math.log(max(0.01, _float(row, key))) for key in
                                       ("gt_size_x_m", "gt_size_y_m", "gt_size_z_m")])
        counts[class_name] += 1
        size = max(_float(row, "gt_bbox_w") * CONTENT_W / 1280.0,
                   _float(row, "gt_bbox_h") * CONTENT_H / 720.0)
        label = "[0,32]" if size <= 32 else "(32,64]" if size <= 64 else "(64,128]" if size <= 128 else \
                "(128,256]" if size <= 256 else "(256,512]" if size <= 512 else ">512"
        projected[class_name][label] += 1
    probabilities = histogram / histogram.sum(axis=1, keepdims=True)
    return {
        "schema": "splitfusion_fcos_train_only_priors_v1",
        "eligible_gt": dict(counts),
        "projected_size_counts": {name: dict(projected[name]) for name in CLASS_NAMES},
        "depth_histogram_with_laplace_one": histogram.astype(int).tolist(),
        "depth_bin_log_probability_bias": np.log(probabilities).tolist(),
        "mean_log_dimensions": {name: np.mean(np.asarray(dimensions[name]), axis=0).tolist() for name in CLASS_NAMES},
        "dense_log1p_bias": 1.440001731467378,
        "semantic_log_frequency_bias": np.log(np.asarray([5364659929, 168777006, 3835646], dtype=np.float64) /
                                                (5364659929 + 168777006 + 3835646)).tolist(),
        "source": "v010 train only; Route B v3.1 camera-plane contract; dense median and pixel counts from audited train cache/contract",
    }

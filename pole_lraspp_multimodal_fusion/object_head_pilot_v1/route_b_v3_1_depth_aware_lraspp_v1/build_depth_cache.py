from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import load_json, read_csv, sha256, utc_now, write_json_x, write_text_x

from data_collection.route_b_perception_v3.visibility_v1 import decode_depth_bgra, depth_is_plausible

MODEL_WIDTH, MODEL_HEIGHT = 768, 432
GRID_WIDTH, GRID_HEIGHT = 192, 108
STRIDE = 4


def source_indices(source_size: int, model_size: int, grid_size: int) -> np.ndarray:
    model_centres = (np.arange(grid_size, dtype=np.float64) + 0.5) * STRIDE
    source_coordinates = model_centres * source_size / model_size
    return np.clip(np.rint(source_coordinates).astype(np.int64), 0, source_size - 1)


def forward_surface(depth_radial: np.ndarray, frame: dict[str, str]) -> np.ndarray:
    height, width = depth_radial.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    fx, fy = float(frame["camera_fx"]), float(frame["camera_fy"])
    cx, cy = float(frame["camera_cx"]), float(frame["camera_cy"])
    ray = np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((cy - v) / fy) ** 2)
    return depth_radial / ray


def radar_matches(frame: dict[str, str], dataset: Path, surface: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    payload = np.load(dataset / frame["radar_points_path"])
    try:
        required = {"world_xyz", "camera_xyz", "camera_depth_m", "u", "v", "valid_projection", "sweep_offset", "observation_age_s"}
        if not required.issubset(payload.files):
            raise RuntimeError(f"radar schema missing {sorted(required - set(payload.files))}")
        camera_xyz = np.asarray(payload["camera_xyz"], dtype=np.float32)
        camera_depth = np.asarray(payload["camera_depth_m"], dtype=np.float32)
        max_depth_delta = float(np.max(np.abs(camera_depth - camera_xyz[:, 0]))) if len(camera_depth) else 0.0
        if max_depth_delta > 1e-4:
            raise RuntimeError(f"camera_depth_m mismatch {max_depth_delta}")
        u = np.asarray(payload["u"], dtype=np.float32)
        v = np.asarray(payload["v"], dtype=np.float32)
        current = (np.asarray(payload["sweep_offset"]) == 0)
        current &= np.isclose(np.asarray(payload["observation_age_s"], dtype=np.float32), 0.0, atol=1e-7)
        current_indices = np.nonzero(current)[0]
        inverse = np.asarray(json.loads(frame["camera_inverse_matrix_json"]), dtype=np.float64)
        world = np.asarray(payload["world_xyz"], dtype=np.float64)[current_indices]
        transformed = (inverse @ np.concatenate([
            world, np.ones((len(world), 1), dtype=np.float64),
        ], axis=1).T).T[:, :3]
        transform_delta = (float(np.max(np.abs(transformed - camera_xyz[current_indices])))
                           if len(current_indices) else 0.0)
        current &= np.asarray(payload["valid_projection"]).astype(bool)
        current &= np.isfinite(camera_depth) & (camera_depth > 0.0)
        current &= (u >= 0.0) & (u < MODEL_WIDTH) & (v >= 0.0) & (v < MODEL_HEIGHT)
        indices = np.nonzero(current)[0]
        if len(indices):
            sx = int(frame["camera_width"]) / MODEL_WIDTH
            sy = int(frame["camera_height"]) / MODEL_HEIGHT
            source_u = np.clip(np.rint(u[indices] * sx).astype(np.int64), 0, surface.shape[1] - 1)
            source_v = np.clip(np.rint(v[indices] * sy).astype(np.int64), 0, surface.shape[0] - 1)
            target = surface[source_v, source_u]
            agree = np.isfinite(target) & (target > 0.0) & (target <= 40.0)
            agree &= np.abs(camera_depth[indices] - target) <= np.maximum(0.5, 0.05 * target)
            indices = indices[agree]
        points = np.empty((len(indices), 3), dtype=np.float32)
        if len(indices):
            points[:, 0] = 2.0 * u[indices] / MODEL_WIDTH - 1.0
            points[:, 1] = 2.0 * v[indices] / MODEL_HEIGHT - 1.0
            points[:, 2] = np.log1p(camera_depth[indices])
        return points, {
            "payload_points": int(len(camera_depth)), "current_valid_points": int(current.sum()),
            "consistent_points": int(len(indices)), "camera_depth_max_abs_delta_m": max_depth_delta,
            "current_sweep_transform_max_abs_delta_m": transform_delta,
        }
    finally:
        payload.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    config = load_json(Path(__file__).with_name("config.json"))
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    dataset = dataset_root / "dataset"
    rows = [row for row in read_csv(dataset / "manifest.csv") if row["split"] == args.split]
    expected = config["data"]["train_frames" if args.split == "train" else "validation_frames"]
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise RuntimeError("cache split population drift")
    if args.split == "val" and not all((experiment / f"predictions/epoch_{epoch:03d}/INFERENCE_COMPLETE").is_file()
                                       for epoch in (10, 20, 30, 40)):
        raise RuntimeError("validation depth cache may be built only after all persisted predictions exist")
    output = experiment / "depth_cache" / args.split
    output.mkdir(parents=True, exist_ok=False)
    depth_path, valid_path = output / "depth_forward_f16.bin", output / "valid_u8.bin"
    radar_path, index_path = output / "radar_consistency_f32.bin", output / "index.csv"
    started = time.monotonic()
    ix = source_indices(int(rows[0]["camera_width"]), MODEL_WIDTH, GRID_WIDTH)
    iy = source_indices(int(rows[0]["camera_height"]), MODEL_HEIGHT, GRID_HEIGHT)
    radar_float_offset = 0
    radar_payload = radar_current = radar_consistent = 0
    max_radar_delta = 0.0
    max_radar_transform_delta = 0.0
    depth_valid_pixels = 0
    metadata_hashes: dict[str, dict[str, str]] = {}
    depth_rows: dict[str, dict[str, dict[str, str]]] = {}
    with depth_path.open("xb") as depth_stream, valid_path.open("xb") as valid_stream, \
            radar_path.open("xb") as radar_stream, index_path.open("x", encoding="utf-8", newline="") as index_stream:
        fields = ("sample_id", "experiment_id", "split", "row_index", "radar_float_offset", "radar_point_count")
        writer = csv.DictWriter(index_stream, fieldnames=fields); writer.writeheader()
        for row_index, frame in enumerate(rows):
            episode = frame["experiment_id"]
            if episode not in metadata_hashes:
                metadata_hashes[episode] = {
                    name: sha256(dataset / episode / name)
                    for name in ("depth_frames.csv", "metadata.json", "resolved_config.json")
                }
                episode_depth_rows = read_csv(dataset / episode / "depth_frames.csv")
                depth_rows[episode] = {value["sample_id"]: value for value in episode_depth_rows}
                if len(depth_rows[episode]) != len(episode_depth_rows):
                    raise RuntimeError(f"duplicate depth synchronization rows in {episode}")
            synchronized = depth_rows[episode].get(frame["sample_id"])
            if synchronized is None:
                raise RuntimeError(f"missing depth synchronization row {frame['sample_id']}")
            if not (synchronized["frame_id"] == frame["frame_id"]
                    == synchronized["rgb_frame_id"] == synchronized["semantic_frame_id"]
                    == synchronized["depth_frame_id"] == synchronized["radar_frame_id"]
                    and float(synchronized["max_timestamp_delta_s"]) == 0.0):
                raise RuntimeError(f"depth synchronization mismatch {frame['sample_id']}")
            png = dataset / episode / "depth" / f"{frame['sample_id']}.png"
            raw = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"missing depth PNG {png}")
            radial = decode_depth_bgra(raw)
            if not depth_is_plausible(radial):
                raise RuntimeError(f"implausible depth frame {png}")
            surface = forward_surface(radial, frame)
            sampled = surface[np.ix_(iy, ix)]
            valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled <= 40.0)
            stored = np.where(valid, sampled, 0.0).astype(np.float16)
            depth_stream.write(stored.tobytes(order="C")); valid_stream.write(valid.astype(np.uint8).tobytes(order="C"))
            points, radar_report = radar_matches(frame, dataset, surface)
            radar_stream.write(points.tobytes(order="C"))
            writer.writerow({
                "sample_id": frame["sample_id"], "experiment_id": episode, "split": args.split,
                "row_index": row_index, "radar_float_offset": radar_float_offset,
                "radar_point_count": len(points),
            })
            radar_float_offset += int(points.size)
            radar_payload += radar_report["payload_points"]
            radar_current += radar_report["current_valid_points"]
            radar_consistent += radar_report["consistent_points"]
            max_radar_delta = max(max_radar_delta, radar_report["camera_depth_max_abs_delta_m"])
            max_radar_transform_delta = max(
                max_radar_transform_delta, radar_report["current_sweep_transform_max_abs_delta_m"],
            )
            depth_valid_pixels += int(valid.sum())
            if (row_index + 1) % 1000 == 0:
                print(f"[depth cache {args.split}] {row_index + 1}/{len(rows)}", flush=True)
        for stream in (depth_stream, valid_stream, radar_stream, index_stream):
            stream.flush(); os.fsync(stream.fileno())
    report = {
        "schema": "route_b_v3_1_depth_aware_cache_v1", "created_utc": utc_now(),
        "split": args.split, "entries": len(rows), "key": "sample_id",
        "depth_shape_per_entry": [GRID_HEIGHT, GRID_WIDTH], "depth_dtype": "float16 camera-forward metres",
        "valid_shape_per_entry": [GRID_HEIGHT, GRID_WIDTH], "valid_dtype": "uint8",
        "depth_sampling": "nearest source pixel to exact stride-4 model cell-centre ray",
        "depth_valid_pixels": depth_valid_pixels,
        "radar_payload_points": radar_payload, "radar_current_valid_points": radar_current,
        "radar_consistent_points": radar_consistent, "radar_camera_depth_max_abs_delta_m": max_radar_delta,
        "radar_current_sweep_transform_max_abs_delta_m": max_radar_transform_delta,
        "depth_synchronization": "all frame IDs equal and maximum timestamp delta exactly zero",
        "files": {name.name: {"bytes": name.stat().st_size, "sha256": sha256(name)}
                  for name in (depth_path, valid_path, radar_path, index_path)},
        "depth_metadata_hashes": metadata_hashes,
        "test_entries": 0, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(output / "CACHE_REPORT.json", report)
    write_text_x(output / "CACHE_COMPLETE", f"{args.split.upper()}_CACHE_COMPLETE\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

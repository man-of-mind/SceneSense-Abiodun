"""Grouped real-CARLA replay registry and 20 Hz GT/prediction resampling."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .types import SceneFrame, SceneObject


@dataclass(frozen=True)
class TraceRecord:
    episode_id: str
    run_group: str
    scenario_family: str
    split: str
    ground_truth_path: Path
    prediction_path: Optional[Path]
    ground_truth_sha256: str
    prediction_sha256: Optional[str]


def normalize_scenario_family(run_group: str) -> str:
    value = run_group.lower().strip()
    value = re.sub(r"^egofpsfast_\d+$", "egofpsfast", value)
    value = re.sub(r"^egofps_\d+$", "egofps", value)
    value = re.sub(r"^fps_\d+$", "pole_fps", value)
    value = re.sub(r"^val_(ae\d+|noae).*", "validation_ego", value)
    value = re.sub(r"^parked_val_(ae\d+|noae).*", "parked_validation", value)
    value = re.sub(r"^speedsweep_fresh_.*", "speedsweep_fresh", value)
    value = re.sub(r"^speedsweep_.*", "speedsweep", value)
    return value


def _family_splits(families: Sequence[str], ratios: Mapping[str, float], seed: int) -> Dict[str, str]:
    unique = sorted(set(families), key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    if len(unique) < 3:
        raise ValueError("at least three scenario families are required for grouped train/validation/test splits")
    n = len(unique)
    n_train = max(1, int(round(n * float(ratios["train"]))))
    n_validation = max(1, int(round(n * float(ratios["validation"]))))
    if n_train + n_validation >= n:
        n_train = max(1, n - 2)
        n_validation = 1
    mapping: Dict[str, str] = {}
    for index, family in enumerate(unique):
        if index < n_train:
            split = "train"
        elif index < n_train + n_validation:
            split = "validation"
        else:
            split = "test"
        mapping[family] = split
    return mapping


def discover_trace_registry(config: Mapping[str, object]) -> List[TraceRecord]:
    spec = config["replay"]
    candidates: List[Tuple[Path, str, str, str, Optional[Path]]] = []
    for root_name in spec["roots"]:
        root = REPO_ROOT / str(root_name)
        for path in sorted(root.glob(str(spec["gt_glob"]))):
            try:
                first = pd.read_csv(path, nrows=1)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if first.empty:
                continue
            required = {"run_id", "run_group", "frame_id", "carla_timestamp", "actor_id", "class_name", "world_x", "world_y", "distance_m"}
            if not required.issubset(first.columns):
                continue
            episode_id = str(first.iloc[0]["run_id"])
            run_group = str(first.iloc[0]["run_group"])
            family = normalize_scenario_family(run_group)
            prediction_name = path.name.replace("_object_ground_truth.csv", str(spec["prediction_suffix"]))
            prediction_path = path.with_name(prediction_name)
            if not prediction_path.exists() or prediction_path.stat().st_size == 0:
                prediction_path = None
            candidates.append((path, episode_id, run_group, family, prediction_path))
    split_map = _family_splits(
        [entry[3] for entry in candidates], spec["split_ratios"], int(spec["split_seed"])
    )
    registry = []
    for path, episode_id, run_group, family, prediction_path in candidates:
        registry.append(
            TraceRecord(
                episode_id=episode_id,
                run_group=run_group,
                scenario_family=family,
                split=split_map[family],
                ground_truth_path=path,
                prediction_path=prediction_path,
                ground_truth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                prediction_sha256=(
                    hashlib.sha256(prediction_path.read_bytes()).hexdigest() if prediction_path else None
                ),
            )
        )
    if not registry:
        raise ValueError("no non-empty replay traces were discovered")
    return registry


def registry_frame(registry: Sequence[TraceRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode_id": item.episode_id,
                "run_group": item.run_group,
                "scenario_family": item.scenario_family,
                "split": item.split,
                "ground_truth_path": str(item.ground_truth_path.relative_to(REPO_ROOT)),
                "prediction_path": (
                    str(item.prediction_path.relative_to(REPO_ROOT)) if item.prediction_path else ""
                ),
                "ground_truth_sha256": item.ground_truth_sha256,
                "prediction_sha256": item.prediction_sha256 or "",
            }
            for item in registry
        ]
    )


def _normalize_class(value: str) -> str:
    value = str(value).lower()
    if value in {"person", "pedestrian", "walker"}:
        return "pedestrian"
    return "vehicle" if "vehicle" in value else value


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _greedy_prediction_matches(gt: pd.DataFrame, predictions: pd.DataFrame, gate_m: float) -> pd.DataFrame:
    matches = []
    for frame_id, gt_frame in gt.groupby("frame_id", sort=False):
        pred_frame = predictions[predictions["frame_id"] == frame_id]
        if pred_frame.empty:
            continue
        pairs = []
        for gt_index, gt_row in gt_frame.iterrows():
            gt_class = _normalize_class(gt_row["class_name"])
            for pred_index, pred_row in pred_frame.iterrows():
                if _normalize_class(pred_row["class_name"]) != gt_class:
                    continue
                distance = float(np.hypot(gt_row["world_x"] - pred_row["world_x"], gt_row["world_y"] - pred_row["world_y"]))
                if distance <= gate_m:
                    pairs.append((distance, gt_index, pred_index))
        used_gt = set()
        used_pred = set()
        for distance, gt_index, pred_index in sorted(pairs):
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            gt_row = gt.loc[gt_index]
            pred_row = predictions.loc[pred_index]
            matches.append(
                {
                    "actor_id": int(gt_row["actor_id"]),
                    "timestamp": float(gt_row["carla_timestamp"]),
                    "world_x": float(pred_row["world_x"]),
                    "world_y": float(pred_row["world_y"]),
                    "range_m": float(pred_row["distance_m"]),
                    "confidence": float(pred_row.get("score", 1.0)),
                    "match_error_m": distance,
                }
            )
    return pd.DataFrame(matches)


def _track_arrays(frame: pd.DataFrame, timestamp_column: str) -> Dict[int, Dict[str, np.ndarray]]:
    tracks: Dict[int, Dict[str, np.ndarray]] = {}
    for actor_id, group in frame.groupby("actor_id"):
        group = group.sort_values(timestamp_column).drop_duplicates(timestamp_column)
        times = group[timestamp_column].to_numpy(dtype=float)
        x = group["world_x"].to_numpy(dtype=float)
        y = group["world_y"].to_numpy(dtype=float)
        if len(times) == 1:
            speed = np.zeros(1, dtype=float)
        else:
            speed = np.hypot(np.gradient(x, times), np.gradient(y, times))
            speed = pd.Series(speed).rolling(3, center=True, min_periods=1).median().to_numpy()
        tracks[int(actor_id)] = {
            "time": times,
            "x": x,
            "y": y,
            "range": group["range_m"].to_numpy(dtype=float),
            "speed": np.maximum(speed, 0.0),
            "confidence": group.get("confidence", pd.Series(np.ones(len(group)))).to_numpy(dtype=float),
            "class": np.array([_normalize_class(group.iloc[0].get("class_name", "vehicle"))]),
        }
    return tracks


def _interpolate(track: Dict[str, np.ndarray], timestamp: float, max_gap_s: float) -> Optional[Dict[str, float]]:
    times = track["time"]
    if timestamp < times[0] or timestamp > times[-1]:
        return None
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right < len(times) and abs(times[right] - timestamp) < 1e-9:
        left = right
    else:
        left = right - 1
    if left < 0 or right >= len(times):
        nearest = min(max(right, 0), len(times) - 1)
        if abs(times[nearest] - timestamp) > max_gap_s:
            return None
    elif right != left and times[right] - times[left] > max_gap_s:
        return None
    return {
        key: float(np.interp(timestamp, times, track[key]))
        for key in ("x", "y", "range", "speed", "confidence")
    }


def load_trace_episode(
    record: TraceRecord,
    config: Mapping[str, object],
    range_m: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> List[SceneFrame]:
    spec = config["replay"]
    hz = float(config["clock"]["hz"])
    dt = 1.0 / hz
    gate = float(range_m if range_m is not None else config["safety"]["range_m"])
    max_gap = float(spec["interpolation_max_gap_s"])
    observation_hold = float(spec.get("observation_track_hold_s", max_gap))
    speed_sigma_floor = float(spec["observation_speed_sigma_floor_mps"])
    gt = pd.read_csv(record.ground_truth_path)
    gt = gt.dropna(subset=["carla_timestamp", "actor_id", "world_x", "world_y", "distance_m"]).copy()
    if bool(spec.get("require_in_camera_frustum", True)):
        if "in_camera_frustum" not in gt.columns:
            raise ValueError(f"missing in_camera_frustum validity field: {record.ground_truth_path}")
        gt = gt[gt["in_camera_frustum"].map(_truthy)].copy()
    if {"origin_x", "origin_y"}.issubset(gt.columns):
        use_origin = gt["origin_x"].notna() & gt["origin_y"].notna()
        gt.loc[use_origin, "world_x"] = gt.loc[use_origin, "origin_x"]
        gt.loc[use_origin, "world_y"] = gt.loc[use_origin, "origin_y"]
    gt = gt.rename(columns={"distance_m": "range_m"})
    gt["class_name"] = gt["class_name"].map(_normalize_class)
    if gt.empty:
        raise ValueError(f"empty ground-truth trace: {record.ground_truth_path}")
    gt_tracks = _track_arrays(gt, "carla_timestamp")
    pred_tracks: Dict[int, Dict[str, np.ndarray]] = {}
    if record.prediction_path is not None:
        predictions = pd.read_csv(record.prediction_path)
        predictions = predictions.dropna(subset=["frame_id", "world_x", "world_y", "distance_m"]).copy()
        predictions = predictions[
            predictions.get("score", pd.Series(1.0, index=predictions.index)).astype(float)
            >= float(spec.get("prediction_score_min", 0.20))
        ].copy()
        matches = _greedy_prediction_matches(gt, predictions, float(spec["association_gate_m"]))
        if not matches.empty:
            matches["class_name"] = "vehicle"
            pred_tracks = _track_arrays(matches, "timestamp")
    start = float(gt["carla_timestamp"].min())
    end = float(gt["carla_timestamp"].max())
    limit = int(max_steps if max_steps is not None else spec["max_episode_steps"])
    count = min(limit, int(np.floor((end - start) / dt)) + 1)
    frames: List[SceneFrame] = []
    for step in range(count):
        absolute_time = start + step * dt
        truth_objects = []
        observed_objects = []
        for actor_id, track in gt_tracks.items():
            values = _interpolate(track, absolute_time, max_gap)
            if values is None or values["range"] > gate:
                continue
            key = (record.episode_id, actor_id)
            truth_objects.append(
                SceneObject(
                    track_key=key,
                    class_name=str(track["class"][0]),
                    world_x=values["x"],
                    world_y=values["y"],
                    range_m=values["range"],
                    speed_mps=values["speed"],
                )
            )
            pred_track = pred_tracks.get(actor_id)
            if pred_track is not None:
                observed = _interpolate(pred_track, absolute_time, observation_hold)
                if observed is not None and observed["range"] <= gate:
                    observed_objects.append(
                        SceneObject(
                            track_key=key,
                            class_name=str(track["class"][0]),
                            world_x=observed["x"],
                            world_y=observed["y"],
                            range_m=observed["range"],
                            speed_mps=observed["speed"],
                            speed_sigma_mps=speed_sigma_floor,
                            confidence=observed["confidence"],
                        )
                    )
        frames.append(
            SceneFrame(
                episode_id=record.episode_id,
                step_index=step,
                timestamp_s=step * dt,
                truth_objects=tuple(truth_objects),
                observed_objects=tuple(observed_objects),
            )
        )
    return frames


def synthetic_episode(
    episode_id: str,
    speeds_mps: Sequence[float],
    steps: int,
    hz: float = 20.0,
    initial_range_m: float = 15.0,
    observed: bool = True,
    class_name: str = "vehicle",
) -> List[SceneFrame]:
    frames = []
    dt = 1.0 / hz
    for step in range(steps):
        objects = []
        for index, speed in enumerate(speeds_mps):
            obj = SceneObject(
                track_key=(episode_id, index + 1),
                class_name=class_name,
                world_x=float(speed) * step * dt,
                world_y=float(index * 3),
                range_m=initial_range_m,
                speed_mps=float(speed),
                speed_sigma_mps=0.25 if observed else 0.0,
                confidence=1.0,
            )
            objects.append(obj)
        frames.append(
            SceneFrame(
                episode_id=episode_id,
                step_index=step,
                timestamp_s=step * dt,
                truth_objects=tuple(objects),
                observed_objects=tuple(objects) if observed else tuple(),
            )
        )
    return frames

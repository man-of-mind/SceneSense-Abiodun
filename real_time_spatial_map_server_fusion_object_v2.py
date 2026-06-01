#!/usr/bin/env python3

"""
Multi-client live top-down spatial map for pole RGB+radar fusion object inference.

This server ingests frame-keyed object-localization results streamed by
multiple ``carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py``
instances, identifies common objects observed by different traffic-light-pole
clients, estimates the per-object localization distribution, and renders a
smoothed fused spatial map over the same road, building, and traffic-light-anchor
map layers used by ``extract_traffic_lights.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import socket
import threading
import time
import zlib
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, request
from matplotlib.patches import Ellipse, Polygon

import extract_traffic_lights as traffic_map


app = Flask(__name__)

DEFAULT_UDP_PORT = 39201
DEFAULT_API_PORT = 35011
SPATIAL_STREAM_SCHEMA = "fusion_object_spatial_map.v1"
FUSED_SPATIAL_MAP_SCHEMA = "fusion_object_spatial_map.fused.v2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OBJECT_COLOR_MAP = {
    "Vehicle": "#00d1ff",
    "Pedestrian": "#ff5fd1",
    "Cyclist": "#8aff80",
    "TrafficLight": "#ff595e",
    "TrafficSign": "#ffd166",
    "Unknown": "#ffffff",
}

CONFIG: Optional[argparse.Namespace] = None
STOP_EVENT = threading.Event()

state_lock = threading.Lock()
static_map_lock = threading.Lock()
render_lock = threading.Lock()
plot_lock = threading.Lock()
fusion_lock = threading.Lock()

latest_streams: Dict[str, Dict[str, object]] = {}
fusion_tracks: Dict[str, Dict[str, object]] = {}
next_track_id = 1
static_map_cache: Dict[str, object] = {
    "loaded_at": 0.0,
    "map_name": None,
    "roads": None,
    "buildings": None,
    "traffic_lights": None,
    "error": None,
}

latest_render_png: Optional[bytes] = None
latest_render_error: Optional[str] = None
latest_render_at = 0.0
latest_written_at = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive fusion-object UDP results from multiple pole clients and "
            "serve a fused live top-down CARLA spatial map with scaled object "
            "footprints and per-object localization distributions."
        )
    )
    parser.add_argument("--api-host", default="0.0.0.0", help="Flask API bind host.")
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="Flask API port.")
    parser.add_argument("--udp-host", default="0.0.0.0", help="UDP ingest bind host.")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="UDP ingest port.")
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument(
        "--render-hz",
        type=float,
        default=4.0,
        help="Top-down map render refresh rate.",
    )
    parser.add_argument(
        "--stream-stale-s",
        type=float,
        default=2.5,
        help="Drop stream detections from the live map after this many silent seconds.",
    )
    parser.add_argument(
        "--anchor-stale-s",
        type=float,
        default=3.0,
        help="Mark a traffic-light anchor inactive after this many silent seconds.",
    )
    parser.add_argument(
        "--static-map-refresh-s",
        type=float,
        default=10.0,
        help="Minimum seconds between CARLA static-map refresh attempts.",
    )
    parser.add_argument(
        "--min-object-score",
        type=float,
        default=0.0,
        help="Suppress detections below this score in the spatial map.",
    )
    parser.add_argument(
        "--max-rendered-objects",
        type=int,
        default=200,
        help="Safety cap for the number of object footprints drawn per render.",
    )
    parser.add_argument(
        "--association-radius-m",
        type=float,
        default=4.0,
        help="Max global XY distance for associating detections from different clients.",
    )
    parser.add_argument(
        "--association-dimension-ratio",
        type=float,
        default=0.65,
        help=(
            "Max relative length/width/height disagreement for common-object "
            "association. Use a larger value for noisier checkpoints."
        ),
    )
    parser.add_argument(
        "--common-min-streams",
        type=int,
        default=2,
        help="Number of distinct client streams required to mark a cluster as common.",
    )
    parser.add_argument(
        "--track-match-radius-m",
        type=float,
        default=6.0,
        help="Max XY distance for matching a current fused cluster to an existing smoothed track.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.45,
        help="EMA measurement weight for fused location, orientation, and dimensions.",
    )
    parser.add_argument(
        "--track-stale-s",
        type=float,
        default=4.0,
        help="Seconds before a smoothed fused track is removed after no observations.",
    )
    parser.add_argument(
        "--hide-single-stream-objects",
        action="store_true",
        help="Only render clusters observed by at least --common-min-streams clients.",
    )
    parser.add_argument(
        "--draw-raw-distribution",
        action="store_true",
        help="Draw faint raw per-client detections around each fused estimate.",
    )
    parser.add_argument(
        "--object-yaw-map-offset-deg",
        type=float,
        default=90.0,
        help=(
            "Yaw offset applied when converting learned object yaw to the "
            "top-down spatial-map footprint convention. The model yaw is "
            "preserved as model_yaw_deg in API responses."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR,
        help="Directory for latest_fusion_object_spatial_map.png.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Also show an OpenCV live window when a graphical display is available.",
    )
    parser.add_argument(
        "--label-inactive-anchors",
        action="store_true",
        help="Annotate inactive traffic-light anchors too. Active anchors are always labeled.",
    )
    parser.add_argument(
        "--focus-traffic-light-ids",
        default="",
        help=(
            "Comma-separated traffic-light actor IDs or OpenDRIVE IDs to zoom "
            "around in the rendered top-down map. Empty keeps the full map."
        ),
    )
    parser.add_argument(
        "--focus-radius-m",
        type=float,
        default=80.0,
        help="Half-width/half-height in meters around each focused traffic-light anchor.",
    )
    parser.add_argument(
        "--focus-padding-m",
        type=float,
        default=10.0,
        help="Additional meters around the focused-anchor union bounds.",
    )
    return parser.parse_args()


def _config() -> argparse.Namespace:
    if CONFIG is None:
        raise RuntimeError("Server configuration has not been initialized.")
    return CONFIG


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _object_map_yaw_deg(model_yaw_deg: float) -> float:
    return _wrap_degrees(
        float(model_yaw_deg) + float(_config().object_yaw_map_offset_deg)
    )


def _focus_traffic_light_ids() -> Set[str]:
    raw_value = str(getattr(_config(), "focus_traffic_light_ids", "") or "")
    return {
        item.strip()
        for item in raw_value.replace(";", ",").split(",")
        if item.strip()
    }


def _anchor_matches_focus(anchor: Dict[str, object], focus_ids: Set[str]) -> bool:
    if not focus_ids:
        return False
    candidates = {str(anchor.get("id", "")).strip()}
    opendrive_id = str(anchor.get("opendrive_id", "") or "").strip()
    if opendrive_id:
        candidates.add(opendrive_id)
    return bool(candidates & focus_ids)


def _focus_view_bounds(anchors: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    focus_ids = _focus_traffic_light_ids()
    if not focus_ids:
        return None

    focus_anchors = [anchor for anchor in anchors if _anchor_matches_focus(anchor, focus_ids)]
    if not focus_anchors:
        return {
            "requested_ids": sorted(focus_ids),
            "matched_ids": [],
            "bounds": None,
            "warning": "No requested focus traffic-light anchors were found in the static map.",
        }

    radius_m = max(1.0, float(_config().focus_radius_m))
    padding_m = max(0.0, float(_config().focus_padding_m))
    xs = []
    ys = []
    matched_ids = []
    for anchor in focus_anchors:
        location = anchor.get("location", {})
        x = _safe_float(location.get("x"))
        y = _safe_float(location.get("y"))
        xs.extend([x - radius_m, x + radius_m])
        ys.extend([y - radius_m, y + radius_m])
        matched_ids.append(str(anchor.get("id")))

    return {
        "requested_ids": sorted(focus_ids),
        "matched_ids": sorted(matched_ids),
        "radius_m": radius_m,
        "padding_m": padding_m,
        "bounds": {
            "x_min": float(min(xs) - padding_m),
            "x_max": float(max(xs) + padding_m),
            "y_min": float(min(ys) - padding_m),
            "y_max": float(max(ys) + padding_m),
        },
        "warning": None,
    }


def _load_traffic_lights_json() -> List[Dict[str, object]]:
    path = os.path.join(SCRIPT_DIR, "traffic_lights_data.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    traffic_lights = []
    for row in rows:
        location = row.get("location", {}) if isinstance(row, dict) else {}
        traffic_lights.append(
            {
                "id": _safe_int(row.get("id"), -1),
                "opendrive_id": str(row.get("opendrive_id", "")),
                "location": {
                    "x": _safe_float(location.get("x")),
                    "y": _safe_float(location.get("y")),
                    "z": _safe_float(location.get("z")),
                },
            }
        )
    return [tl for tl in traffic_lights if int(tl["id"]) >= 0]


def _traffic_light_opendrive_id(actor: "traffic_map.carla.Actor") -> str:
    try:
        value = actor.get_opendrive_id()
    except Exception:
        return ""
    return "" if value is None else str(value)


def _refresh_static_map_context() -> Dict[str, object]:
    cfg = _config()
    now = time.time()
    with static_map_lock:
        cache_ready = (
            static_map_cache.get("roads") is not None
            and static_map_cache.get("buildings") is not None
            and static_map_cache.get("traffic_lights") is not None
        )
        if cache_ready and now - float(static_map_cache["loaded_at"]) < float(cfg.static_map_refresh_s):
            return dict(static_map_cache)

        try:
            client = traffic_map.carla.Client(str(cfg.carla_host), int(cfg.carla_port))
            client.set_timeout(10.0)
            world = client.get_world()
            carla_map = world.get_map()
            roads, buildings = traffic_map._build_precise_static_map(world, carla_map)

            traffic_lights = []
            actors = sorted(world.get_actors().filter("traffic.traffic_light"), key=lambda actor: actor.id)
            for actor in actors:
                location = actor.get_location()
                traffic_lights.append(
                    {
                        "id": int(actor.id),
                        "opendrive_id": _traffic_light_opendrive_id(actor),
                        "location": {
                            "x": float(location.x),
                            "y": float(location.y),
                            "z": float(location.z),
                        },
                    }
                )

            static_map_cache.update(
                {
                    "loaded_at": now,
                    "map_name": carla_map.name,
                    "roads": roads,
                    "buildings": buildings,
                    "traffic_lights": traffic_lights,
                    "error": None,
                }
            )
        except Exception as exc:
            if cache_ready:
                static_map_cache["loaded_at"] = now
                static_map_cache["error"] = str(exc)
                return dict(static_map_cache)

            fallback_anchors = _load_traffic_lights_json()
            static_map_cache.update(
                {
                    "loaded_at": now,
                    "map_name": "CARLA",
                    "roads": [],
                    "buildings": [],
                    "traffic_lights": fallback_anchors,
                    "error": str(exc),
                }
            )

        return dict(static_map_cache)


def _normalize_object(
    obj: Dict[str, object],
    *,
    stream_id: str,
    frame_id: int,
    index: int,
) -> Optional[Dict[str, object]]:
    location = obj.get("location") if isinstance(obj.get("location"), dict) else {}
    dimensions = obj.get("dimensions") if isinstance(obj.get("dimensions"), dict) else {}
    score = _safe_float(obj.get("score"), 0.0)
    if score < float(_config().min_object_score):
        return None

    object_type = str(obj.get("type") or "Unknown")
    motion_state = str(obj.get("motion_state") or "")
    if object_type == "ParkedVehicle":
        object_type = "Vehicle"
        motion_state = motion_state or "parked"
    elif object_type == "MovingVehicle":
        object_type = "Vehicle"
        motion_state = motion_state or "moving"

    model_yaw_deg = _safe_float(obj.get("model_yaw_deg", obj.get("yaw_deg")), 0.0)

    return {
        "id": str(obj.get("id") or f"{stream_id}:{frame_id}:{index}"),
        "source_stream_id": stream_id,
        "frame_id": int(frame_id),
        "type": object_type,
        "motion_state": motion_state,
        "score": score,
        "location": {
            "x": _safe_float(location.get("x", obj.get("world_x"))),
            "y": _safe_float(location.get("y", obj.get("world_y"))),
            "z": _safe_float(location.get("z", obj.get("world_z"))),
        },
        "dimensions": {
            "length": max(0.05, _safe_float(dimensions.get("length", obj.get("size_x")), 0.05)),
            "width": max(0.05, _safe_float(dimensions.get("width", obj.get("size_y")), 0.05)),
            "height": max(0.05, _safe_float(dimensions.get("height", obj.get("size_z")), 0.05)),
        },
        "yaw_deg": _object_map_yaw_deg(model_yaw_deg),
        "model_yaw_deg": model_yaw_deg,
        "map_yaw_offset_deg": float(_config().object_yaw_map_offset_deg),
        "parked_score": _safe_float(obj.get("parked_score"), 0.0),
        "radar_support_score": _safe_float(obj.get("radar_support_score"), 0.0),
        "bbox_xyxy": obj.get("bbox_xyxy"),
    }


def _normalize_packet(payload: Dict[str, object], received_at: float) -> Dict[str, object]:
    stream_id = str(payload.get("stream_id") or payload.get("node_id") or "fusion_stream")
    frame_id = _safe_int(payload.get("frame_id"), 0)
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raw_objects = payload.get("detections")
    if not isinstance(raw_objects, list):
        raw_objects = []

    objects = []
    for index, obj in enumerate(raw_objects):
        if not isinstance(obj, dict):
            continue
        normalized = _normalize_object(obj, stream_id=stream_id, frame_id=frame_id, index=index)
        if normalized is not None:
            objects.append(normalized)

    return {
        "schema": str(payload.get("schema") or ""),
        "stream_id": stream_id,
        "node_id": str(payload.get("node_id") or stream_id),
        "traffic_light_id": str(payload.get("traffic_light_id") or ""),
        "traffic_light_actor_id": _safe_int(payload.get("traffic_light_actor_id"), -1),
        "traffic_light_opendrive_id": str(payload.get("traffic_light_opendrive_id") or ""),
        "frame_id": frame_id,
        "timestamp": _safe_float(payload.get("timestamp"), received_at),
        "carla_timestamp": _safe_float(payload.get("carla_timestamp"), 0.0),
        "received_at": received_at,
        "camera": payload.get("camera") if isinstance(payload.get("camera"), dict) else {},
        "segmentation": payload.get("segmentation") if isinstance(payload.get("segmentation"), dict) else {},
        "latency": payload.get("latency") if isinstance(payload.get("latency"), dict) else {},
        "objects": objects,
        "object_count": len(objects),
        "source_script": str(payload.get("source_script") or ""),
    }


def _object_xy(obj: Dict[str, object]) -> np.ndarray:
    location = obj.get("location", {})
    return np.array(
        [_safe_float(location.get("x")), _safe_float(location.get("y"))],
        dtype=np.float64,
    )


def _object_xyz(obj: Dict[str, object]) -> np.ndarray:
    location = obj.get("location", {})
    return np.array(
        [
            _safe_float(location.get("x")),
            _safe_float(location.get("y")),
            _safe_float(location.get("z")),
        ],
        dtype=np.float64,
    )


def _object_dimensions(obj: Dict[str, object]) -> np.ndarray:
    dimensions = obj.get("dimensions", {})
    return np.array(
        [
            max(0.05, _safe_float(dimensions.get("length"), 0.05)),
            max(0.05, _safe_float(dimensions.get("width"), 0.05)),
            max(0.05, _safe_float(dimensions.get("height"), 0.05)),
        ],
        dtype=np.float64,
    )


def _score_weight(obj: Dict[str, object]) -> float:
    score = max(0.05, _safe_float(obj.get("score"), 0.0))
    radar_support = max(0.0, _safe_float(obj.get("radar_support_score"), 0.0))
    return float(score * (1.0 + 0.5 * min(1.0, radar_support)))


def _dimension_compatible(a: Dict[str, object], b: Dict[str, object]) -> bool:
    threshold = max(0.0, float(_config().association_dimension_ratio))
    dims_a = _object_dimensions(a)
    dims_b = _object_dimensions(b)
    denom = np.maximum(np.maximum(dims_a, dims_b), 0.1)
    return bool(np.max(np.abs(dims_a - dims_b) / denom) <= threshold)


def _circular_mean_deg(yaws_deg: np.ndarray, weights: np.ndarray) -> float:
    if yaws_deg.size == 0:
        return 0.0
    radians = np.deg2rad(yaws_deg.astype(np.float64))
    sin_sum = float(np.sum(np.sin(radians) * weights))
    cos_sum = float(np.sum(np.cos(radians) * weights))
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return float(yaws_deg[-1])
    return float(math.degrees(math.atan2(sin_sum, cos_sum)))


def _circular_std_deg(yaws_deg: np.ndarray, weights: np.ndarray) -> float:
    if yaws_deg.size <= 1:
        return 0.0
    weights = weights / max(1e-9, float(np.sum(weights)))
    radians = np.deg2rad(yaws_deg.astype(np.float64))
    r = math.hypot(
        float(np.sum(np.cos(radians) * weights)),
        float(np.sum(np.sin(radians) * weights)),
    )
    r = min(1.0, max(1e-9, r))
    return float(math.degrees(math.sqrt(max(0.0, -2.0 * math.log(r)))))


def _blend_yaw_deg(previous_deg: float, measurement_deg: float, alpha: float) -> float:
    return _circular_mean_deg(
        np.array([previous_deg, measurement_deg], dtype=np.float64),
        np.array([1.0 - alpha, alpha], dtype=np.float64),
    )


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    weights = weights.astype(np.float64)
    weight_sum = max(1e-9, float(np.sum(weights)))
    normalized = weights / weight_sum
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum(((values - mean) ** 2) * normalized[:, None], axis=0)
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _raw_observation_payload(obj: Dict[str, object]) -> Dict[str, object]:
    location = obj.get("location", {})
    dimensions = obj.get("dimensions", {})
    return {
        "id": obj.get("id"),
        "source_stream_id": obj.get("source_stream_id"),
        "frame_id": obj.get("frame_id"),
        "type": obj.get("type"),
        "motion_state": obj.get("motion_state"),
        "score": _safe_float(obj.get("score"), 0.0),
        "location": {
            "x": _safe_float(location.get("x")),
            "y": _safe_float(location.get("y")),
            "z": _safe_float(location.get("z")),
        },
        "dimensions": {
            "length": max(0.05, _safe_float(dimensions.get("length"), 0.05)),
            "width": max(0.05, _safe_float(dimensions.get("width"), 0.05)),
            "height": max(0.05, _safe_float(dimensions.get("height"), 0.05)),
        },
        "yaw_deg": _safe_float(obj.get("yaw_deg"), 0.0),
        "model_yaw_deg": _safe_float(obj.get("model_yaw_deg", obj.get("yaw_deg")), 0.0),
        "map_yaw_offset_deg": _safe_float(obj.get("map_yaw_offset_deg"), 0.0),
        "parked_score": _safe_float(obj.get("parked_score"), 0.0),
        "radar_support_score": _safe_float(obj.get("radar_support_score"), 0.0),
    }


def _cluster_signature(cluster: Sequence[Dict[str, object]]) -> Tuple[Tuple[str, str, int], ...]:
    return tuple(
        sorted(
            (
                str(obj.get("source_stream_id")),
                str(obj.get("id")),
                _safe_int(obj.get("frame_id"), 0),
            )
            for obj in cluster
        )
    )


def _cluster_detection_distance(cluster: Sequence[Dict[str, object]], obj: Dict[str, object]) -> float:
    cluster_xy = np.vstack([_object_xy(item) for item in cluster])
    center_xy = np.mean(cluster_xy, axis=0)
    return float(np.linalg.norm(_object_xy(obj) - center_xy))


def _can_join_cluster(cluster: Sequence[Dict[str, object]], obj: Dict[str, object]) -> bool:
    source_stream_id = str(obj.get("source_stream_id"))
    if any(str(item.get("source_stream_id")) == source_stream_id for item in cluster):
        return False
    obj_type = str(obj.get("type") or "Unknown")
    if any(str(item.get("type") or "Unknown") != obj_type for item in cluster):
        return False
    if _cluster_detection_distance(cluster, obj) > float(_config().association_radius_m):
        return False
    return all(_dimension_compatible(item, obj) for item in cluster)


def _build_detection_clusters(objects: Sequence[Dict[str, object]]) -> List[List[Dict[str, object]]]:
    clusters: List[List[Dict[str, object]]] = []
    ordered = sorted(
        objects,
        key=lambda item: (_safe_float(item.get("score"), 0.0), -_safe_int(item.get("frame_id"), 0)),
        reverse=True,
    )
    for obj in ordered:
        best_index = None
        best_distance = float("inf")
        for index, cluster in enumerate(clusters):
            if not _can_join_cluster(cluster, obj):
                continue
            distance = _cluster_detection_distance(cluster, obj)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            clusters.append([obj])
        else:
            clusters[best_index].append(obj)
    return clusters


def _cluster_to_measurement(cluster: Sequence[Dict[str, object]]) -> Dict[str, object]:
    observations = [_raw_observation_payload(obj) for obj in cluster]
    locations = np.vstack([_object_xyz(obj) for obj in cluster])
    dimensions = np.vstack([_object_dimensions(obj) for obj in cluster])
    yaws = np.array([_safe_float(obj.get("yaw_deg"), 0.0) for obj in cluster], dtype=np.float64)
    scores = np.array([_safe_float(obj.get("score"), 0.0) for obj in cluster], dtype=np.float64)
    weights = np.array([_score_weight(obj) for obj in cluster], dtype=np.float64)

    location_mean, location_std = _weighted_mean_std(locations, weights)
    dimension_mean, dimension_std = _weighted_mean_std(dimensions, weights)
    yaw_mean = _circular_mean_deg(yaws, weights)
    yaw_std = _circular_std_deg(yaws, weights)
    source_stream_ids = sorted({str(obj.get("source_stream_id")) for obj in cluster})
    support_count = len(source_stream_ids)
    common_min_streams = max(1, int(_config().common_min_streams))
    covariance_xy = np.zeros((2, 2), dtype=np.float64)
    if len(cluster) > 1:
        normalized_weights = weights / max(1e-9, float(np.sum(weights)))
        centered_xy = locations[:, :2] - location_mean[:2]
        covariance_xy = centered_xy.T @ (centered_xy * normalized_weights[:, None])

    motion_votes = Counter(str(obj.get("motion_state") or "unspecified") for obj in cluster)
    object_type = Counter(str(obj.get("type") or "Unknown") for obj in cluster).most_common(1)[0][0]
    standard_error = location_std / math.sqrt(max(1, len(cluster)))
    signature = _cluster_signature(cluster)

    return {
        "track_id": "",
        "type": object_type,
        "motion_state": motion_votes.most_common(1)[0][0],
        "score": float(np.average(scores, weights=weights)),
        "support_stream_count": support_count,
        "source_stream_ids": source_stream_ids,
        "is_common_object": support_count >= common_min_streams,
        "location": {
            "x": float(location_mean[0]),
            "y": float(location_mean[1]),
            "z": float(location_mean[2]),
        },
        "dimensions": {
            "length": float(dimension_mean[0]),
            "width": float(dimension_mean[1]),
            "height": float(dimension_mean[2]),
        },
        "yaw_deg": float(yaw_mean),
        "distribution": {
            "observation_count": len(observations),
            "source_stream_count": support_count,
            "source_stream_ids": source_stream_ids,
            "location_mean": {
                "x": float(location_mean[0]),
                "y": float(location_mean[1]),
                "z": float(location_mean[2]),
            },
            "location_std": {
                "x": float(location_std[0]),
                "y": float(location_std[1]),
                "z": float(location_std[2]),
            },
            "location_standard_error": {
                "x": float(standard_error[0]),
                "y": float(standard_error[1]),
                "z": float(standard_error[2]),
            },
            "xy_covariance": covariance_xy.tolist(),
            "dimensions_mean": {
                "length": float(dimension_mean[0]),
                "width": float(dimension_mean[1]),
                "height": float(dimension_mean[2]),
            },
            "dimensions_std": {
                "length": float(dimension_std[0]),
                "width": float(dimension_std[1]),
                "height": float(dimension_std[2]),
            },
            "yaw_mean_deg": float(yaw_mean),
            "yaw_std_deg": float(yaw_std),
            "raw_observations": observations,
        },
        "measurement_signature": signature,
        "last_observed_at": max(_safe_float(obj.get("received_at"), time.time()) for obj in cluster),
    }


def _track_xy(track: Dict[str, object]) -> np.ndarray:
    location = track.get("location", {})
    return np.array([_safe_float(location.get("x")), _safe_float(location.get("y"))], dtype=np.float64)


def _measurement_xy(measurement: Dict[str, object]) -> np.ndarray:
    location = measurement.get("location", {})
    return np.array([_safe_float(location.get("x")), _safe_float(location.get("y"))], dtype=np.float64)


def _new_track_id() -> str:
    global next_track_id
    track_id = f"fused_object_{next_track_id:04d}"
    next_track_id += 1
    return track_id


def _match_existing_track(
    measurement: Dict[str, object],
    used_track_ids: Set[str],
) -> Optional[str]:
    best_track_id = None
    best_distance = float("inf")
    measurement_type = str(measurement.get("type") or "Unknown")
    measurement_xy = _measurement_xy(measurement)
    for track_id, track in fusion_tracks.items():
        if track_id in used_track_ids:
            continue
        if str(track.get("type") or "Unknown") != measurement_type:
            continue
        distance = float(np.linalg.norm(_track_xy(track) - measurement_xy))
        if distance <= float(_config().track_match_radius_m) and distance < best_distance:
            best_track_id = track_id
            best_distance = distance
    return best_track_id


def _smooth_track(track: Dict[str, object], measurement: Dict[str, object]) -> Dict[str, object]:
    signature = measurement.get("measurement_signature")
    if track.get("measurement_signature") == signature:
        track.update(
            {
                "support_stream_count": measurement["support_stream_count"],
                "source_stream_ids": measurement["source_stream_ids"],
                "is_common_object": measurement["is_common_object"],
                "score": measurement["score"],
                "distribution": measurement["distribution"],
            }
        )
        return track

    alpha = min(1.0, max(0.0, float(_config().smoothing_alpha)))
    old_loc = _object_xyz(track)
    new_loc = _object_xyz(measurement)
    old_dims = _object_dimensions(track)
    new_dims = _object_dimensions(measurement)
    blended_loc = (1.0 - alpha) * old_loc + alpha * new_loc
    blended_dims = (1.0 - alpha) * old_dims + alpha * new_dims
    blended_yaw = _blend_yaw_deg(
        _safe_float(track.get("yaw_deg"), 0.0),
        _safe_float(measurement.get("yaw_deg"), 0.0),
        alpha,
    )

    track.update(
        {
            "type": measurement["type"],
            "motion_state": measurement["motion_state"],
            "score": measurement["score"],
            "support_stream_count": measurement["support_stream_count"],
            "source_stream_ids": measurement["source_stream_ids"],
            "is_common_object": measurement["is_common_object"],
            "location": {
                "x": float(blended_loc[0]),
                "y": float(blended_loc[1]),
                "z": float(blended_loc[2]),
            },
            "dimensions": {
                "length": float(blended_dims[0]),
                "width": float(blended_dims[1]),
                "height": float(blended_dims[2]),
            },
            "yaw_deg": float(blended_yaw),
            "distribution": measurement["distribution"],
            "measurement_signature": signature,
            "last_observed_at": measurement["last_observed_at"],
            "updated_at": time.time(),
        }
    )
    return track


def _fuse_and_smooth_objects(objects: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    global fusion_tracks
    now = time.time()
    clusters = _build_detection_clusters(objects)
    measurements = [_cluster_to_measurement(cluster) for cluster in clusters]

    with fusion_lock:
        stale_track_ids = [
            track_id for track_id, track in fusion_tracks.items()
            if now - _safe_float(track.get("last_observed_at"), now) > float(_config().track_stale_s)
        ]
        for track_id in stale_track_ids:
            fusion_tracks.pop(track_id, None)

        used_track_ids: Set[str] = set()
        fused_objects = []
        for measurement in sorted(
            measurements,
            key=lambda item: (item["is_common_object"], item["support_stream_count"], item["score"]),
            reverse=True,
        ):
            track_id = _match_existing_track(measurement, used_track_ids)
            if track_id is None:
                track_id = _new_track_id()
                measurement["track_id"] = track_id
                measurement["updated_at"] = now
                fusion_tracks[track_id] = measurement
            else:
                measurement["track_id"] = track_id
                fusion_tracks[track_id] = _smooth_track(fusion_tracks[track_id], measurement)
            used_track_ids.add(track_id)
            fused_objects.append(dict(fusion_tracks[track_id]))

    if bool(_config().hide_single_stream_objects):
        fused_objects = [obj for obj in fused_objects if bool(obj.get("is_common_object"))]

    fused_objects = fused_objects[: max(0, int(_config().max_rendered_objects))]
    return fused_objects, measurements


def _active_anchor_keys(streams: Sequence[Dict[str, object]], now: float) -> set:
    cfg = _config()
    active = set()
    for stream in streams:
        if now - float(stream["received_at"]) > float(cfg.anchor_stale_s):
            continue
        for key in ("traffic_light_id", "traffic_light_actor_id", "traffic_light_opendrive_id"):
            value = stream.get(key)
            if value not in (None, "", -1):
                active.add(str(value))
    return active


def _build_spatial_map_snapshot() -> Dict[str, object]:
    cfg = _config()
    now = time.time()
    static_context = _refresh_static_map_context()

    with state_lock:
        streams = [dict(stream) for stream in latest_streams.values()]

    active_streams = []
    stale_streams = []
    raw_objects = []
    for stream in sorted(streams, key=lambda item: str(item["stream_id"])):
        age_s = now - float(stream["received_at"])
        stream_info = {
            "stream_id": stream["stream_id"],
            "node_id": stream["node_id"],
            "traffic_light_id": stream["traffic_light_id"],
            "traffic_light_actor_id": stream["traffic_light_actor_id"],
            "traffic_light_opendrive_id": stream["traffic_light_opendrive_id"],
            "frame_id": stream["frame_id"],
            "age_s": age_s,
            "object_count": stream["object_count"],
            "segmentation": stream["segmentation"],
            "latency": stream["latency"],
            "source_script": stream["source_script"],
        }
        if age_s <= float(cfg.stream_stale_s):
            active_streams.append(stream_info)
            for obj in stream["objects"]:
                enriched = dict(obj)
                enriched["received_at"] = float(stream["received_at"])
                raw_objects.append(enriched)
        else:
            stale_streams.append(stream_info)

    raw_objects = raw_objects[: max(0, int(cfg.max_rendered_objects))]
    fused_objects, object_associations = _fuse_and_smooth_objects(raw_objects)
    active_keys = _active_anchor_keys(streams, now)
    focus_ids = _focus_traffic_light_ids()
    anchors = []
    for anchor in static_context.get("traffic_lights") or []:
        anchor_id = str(anchor.get("id"))
        opendrive_id = str(anchor.get("opendrive_id") or "")
        active = anchor_id in active_keys or (opendrive_id and opendrive_id in active_keys)
        focused = _anchor_matches_focus(anchor, focus_ids)
        anchors.append({**anchor, "active": bool(active), "focus": bool(focused)})
    focus_view = _focus_view_bounds(anchors)

    latest_frame = max((int(stream["frame_id"]) for stream in active_streams), default=None)
    return {
        "frame_id": latest_frame,
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "status": "active" if active_streams else "waiting_for_fusion_results",
        "spatial_map_objects": fused_objects,
        "raw_spatial_map_objects": raw_objects,
        "object_associations": object_associations,
        "traffic_light_anchors": anchors,
        "active_streams": active_streams,
        "stale_streams": stale_streams,
        "metadata": {
            "schema": FUSED_SPATIAL_MAP_SCHEMA,
            "input_schema": SPATIAL_STREAM_SCHEMA,
            "coordinate_system": "global_carla_world",
            "map_name": static_context.get("map_name"),
            "static_map_error": static_context.get("error"),
            "udp_port": int(cfg.udp_port),
            "object_source": "learned_rgb_radar_fusion_head",
            "fusion_policy": {
                "association_radius_m": float(cfg.association_radius_m),
                "association_dimension_ratio": float(cfg.association_dimension_ratio),
                "common_min_streams": int(cfg.common_min_streams),
                "track_match_radius_m": float(cfg.track_match_radius_m),
                "smoothing_alpha": float(cfg.smoothing_alpha),
                "hide_single_stream_objects": bool(cfg.hide_single_stream_objects),
                "object_yaw_map_offset_deg": float(cfg.object_yaw_map_offset_deg),
            },
            "focus_view": focus_view,
            "visualization": {
                "latest_png": "/api/spatial_map/live.png",
                "viewer": "/api/spatial_map/viewer",
            },
        },
    }


def _oriented_box_corners(
    x: float,
    y: float,
    length: float,
    width: float,
    yaw_deg: float,
) -> List[Tuple[float, float]]:
    half_l = max(0.05, float(length)) / 2.0
    half_w = max(0.05, float(width)) / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    yaw = math.radians(float(yaw_deg))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    rotated = local @ rotation.T
    rotated[:, 0] += float(x)
    rotated[:, 1] += float(y)
    return [(float(px), float(py)) for px, py in rotated]


def _draw_static_map(ax: "plt.Axes", static_context: Dict[str, object], anchors: Sequence[Dict[str, object]]) -> None:
    buildings = static_context.get("buildings") or []
    roads = static_context.get("roads") or []
    for building in buildings:
        footprint = building.get("footprint") if isinstance(building, dict) else None
        if not footprint:
            continue
        poly = Polygon(
            [(point["x"], point["y"]) for point in footprint],
            closed=True,
            facecolor="#2a2a2a",
            edgecolor="#404040",
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(poly)

    for polyline in roads:
        if len(polyline) < 2:
            continue
        draw_polyline = traffic_map._smooth_polyline_for_plot(
            polyline,
            traffic_map.ROAD_CENTERLINE_SMOOTHING_PASSES,
        )
        ax.plot(
            [point[0] for point in draw_polyline],
            [point[1] for point in draw_polyline],
            color="#555555",
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
            solid_joinstyle="round",
            solid_capstyle="round",
        )

    inactive_x = []
    inactive_y = []
    active_x = []
    active_y = []
    focus_x = []
    focus_y = []
    for anchor in anchors:
        location = anchor.get("location", {})
        if anchor.get("focus"):
            focus_x.append(_safe_float(location.get("x")))
            focus_y.append(_safe_float(location.get("y")))
        if anchor.get("active"):
            active_x.append(_safe_float(location.get("x")))
            active_y.append(_safe_float(location.get("y")))
        else:
            inactive_x.append(_safe_float(location.get("x")))
            inactive_y.append(_safe_float(location.get("y")))

    if inactive_x:
        ax.scatter(
            inactive_x,
            inactive_y,
            c="#8a8f98",
            s=32,
            marker="^",
            edgecolors="#20242a",
            linewidths=0.5,
            label="Inactive traffic-light anchors",
            alpha=0.65,
            zorder=4,
        )
    if active_x:
        ax.scatter(
            active_x,
            active_y,
            c="#ff3b30",
            s=140,
            marker="^",
            edgecolors="white",
            linewidths=1.0,
            label="Active traffic-light anchors",
            zorder=7,
        )
    if focus_x:
        ax.scatter(
            focus_x,
            focus_y,
            facecolors="none",
            edgecolors="#ffd166",
            s=260,
            marker="o",
            linewidths=1.4,
            label="Focused traffic-light anchors",
            zorder=9,
        )

    label_inactive = bool(_config().label_inactive_anchors)
    for anchor in anchors:
        if not anchor.get("active") and not anchor.get("focus") and not label_inactive:
            continue
        location = anchor.get("location", {})
        color = "#ffd166" if anchor.get("focus") else ("#ff9aa2" if anchor.get("active") else "#a7adb8")
        ax.annotate(
            f"TL {anchor.get('id')}",
            (_safe_float(location.get("x")), _safe_float(location.get("y"))),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            zorder=8,
        )


def _draw_objects(ax: "plt.Axes", objects: Sequence[Dict[str, object]]) -> None:
    for obj in objects:
        location = obj.get("location", {})
        dimensions = obj.get("dimensions", {})
        obj_type = str(obj.get("type") or "Unknown")
        color = OBJECT_COLOR_MAP.get(obj_type, OBJECT_COLOR_MAP["Unknown"])
        x = _safe_float(location.get("x"))
        y = _safe_float(location.get("y"))
        length = _safe_float(dimensions.get("length"), 0.05)
        width = _safe_float(dimensions.get("width"), 0.05)
        yaw_deg = _safe_float(obj.get("yaw_deg"), 0.0)
        corners = _oriented_box_corners(x, y, length, width, yaw_deg)

        poly = Polygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.58,
            linestyle="-" if obj.get("is_common_object") else "--",
            zorder=6,
        )
        ax.add_patch(poly)
        ax.scatter([x], [y], c=color, s=22, edgecolors="white", linewidths=0.5, zorder=7)


def _draw_raw_distribution(ax: "plt.Axes", objects: Sequence[Dict[str, object]]) -> None:
    for obj in objects:
        fused_location = obj.get("location", {})
        fx = _safe_float(fused_location.get("x"))
        fy = _safe_float(fused_location.get("y"))
        distribution = obj.get("distribution", {})
        observations = distribution.get("raw_observations", []) if isinstance(distribution, dict) else []
        if not observations:
            continue

        raw_x = []
        raw_y = []
        for obs in observations:
            location = obs.get("location", {}) if isinstance(obs, dict) else {}
            ox = _safe_float(location.get("x"))
            oy = _safe_float(location.get("y"))
            raw_x.append(ox)
            raw_y.append(oy)
            ax.plot([ox, fx], [oy, fy], color="#d8dee9", alpha=0.18, linewidth=0.8, zorder=5)

        ax.scatter(
            raw_x,
            raw_y,
            c="#d8dee9",
            s=20,
            marker="x",
            alpha=0.65,
            linewidths=0.9,
            zorder=7,
        )

        covariance = distribution.get("xy_covariance") if isinstance(distribution, dict) else None
        if not covariance or len(observations) < 2:
            continue
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (2, 2):
            continue
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 0.0)
        order = values.argsort()[::-1]
        values = values[order]
        vectors = vectors[:, order]
        angle = math.degrees(math.atan2(float(vectors[1, 0]), float(vectors[0, 0])))
        ellipse = Ellipse(
            (fx, fy),
            width=2.0 * math.sqrt(float(values[0])),
            height=2.0 * math.sqrt(float(values[1])),
            angle=angle,
            fill=False,
            edgecolor="#ffffff",
            linewidth=0.8,
            linestyle=":",
            alpha=0.65,
            zorder=7,
        )
        ax.add_patch(ellipse)


def _apply_focus_view(ax: "plt.Axes", snapshot: Dict[str, object]) -> bool:
    metadata = snapshot.get("metadata", {})
    focus_view = metadata.get("focus_view") if isinstance(metadata, dict) else None
    if not isinstance(focus_view, dict):
        return False
    bounds = focus_view.get("bounds")
    if not isinstance(bounds, dict):
        return False

    x_min = _safe_float(bounds.get("x_min"))
    x_max = _safe_float(bounds.get("x_max"))
    y_min = _safe_float(bounds.get("y_min"))
    y_max = _safe_float(bounds.get("y_max"))
    if x_max <= x_min or y_max <= y_min:
        return False

    ax.set_xlim(x_min, x_max)
    # The CARLA top-down view is rendered with an inverted Y axis.
    ax.set_ylim(y_max, y_min)
    return True


def _render_snapshot(snapshot: Dict[str, object]) -> Tuple[Optional[bytes], Optional[str]]:
    static_context = _refresh_static_map_context()
    anchors = snapshot.get("traffic_light_anchors") or []
    objects = snapshot.get("spatial_map_objects") or []
    frame_id = snapshot.get("frame_id")

    try:
        with plot_lock:
            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(15, 15))
                _draw_static_map(ax, static_context, anchors)
                if bool(_config().draw_raw_distribution):
                    _draw_raw_distribution(ax, objects)
                _draw_objects(ax, objects)

                if not objects:
                    ax.text(
                        0.5,
                        0.02,
                        "No live fusion-object detections in the current spatial map",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        color="white",
                    )
                if static_context.get("error"):
                    ax.text(
                        0.5,
                        0.97,
                        f"Static-map warning: {static_context['error']}",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="#ffd166",
                    )
                focus_view = snapshot.get("metadata", {}).get("focus_view", {})
                if isinstance(focus_view, dict) and focus_view.get("warning"):
                    ax.text(
                        0.5,
                        0.93,
                        str(focus_view["warning"]),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=9,
                        color="#ffd166",
                    )

                map_label = static_context.get("map_name") or "CARLA"
                frame_label = "waiting" if frame_id is None else str(frame_id)
                ax.set_title(
                    f"{map_label} - Multi-Client Fused Object Spatial Map (Frame {frame_label})",
                    fontsize=18,
                    pad=20,
                )
                ax.set_xlabel("Global X Coordinate (meters)", fontsize=14)
                ax.set_ylabel("Global Y Coordinate (meters)", fontsize=14)
                ax.set_aspect("equal", adjustable="box")
                if not _apply_focus_view(ax, snapshot):
                    ax.invert_yaxis()
                ax.grid(True, linestyle="--", alpha=0.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)

                img_io = io.BytesIO()
                fig.savefig(img_io, format="png", dpi=160, bbox_inches="tight")
                plt.close(fig)
                img_io.seek(0)
                return img_io.getvalue(), None
    except Exception as exc:
        return None, str(exc)


def _write_latest_png(png_bytes: bytes) -> None:
    global latest_written_at
    now = time.time()
    if now - latest_written_at < 1.0:
        return
    latest_written_at = now
    cfg = _config()
    os.makedirs(str(cfg.output_dir), exist_ok=True)
    output_path = os.path.join(str(cfg.output_dir), "latest_fusion_object_spatial_map_v2.png")
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(png_bytes)
        os.replace(tmp_path, output_path)
    except OSError as exc:
        print(f"[Render] Failed writing latest PNG: {exc}")


def _has_graphical_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _maybe_show_display(png_bytes: bytes) -> None:
    cfg = _config()
    if not bool(cfg.display) or not _has_graphical_display():
        return
    try:
        import cv2

        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        cv2.imshow("Live Fusion Object Spatial Map", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            STOP_EVENT.set()
    except Exception as exc:
        print(f"[Display] OpenCV display disabled after error: {exc}")
        cfg.display = False


def udp_listener_thread() -> None:
    cfg = _config()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((str(cfg.udp_host), int(cfg.udp_port)))
    sock.settimeout(0.5)
    print(f"[UDP] Listening for fusion-object spatial packets on {cfg.udp_host}:{cfg.udp_port}")

    while not STOP_EVENT.is_set():
        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        received_at = time.time()
        try:
            payload = json.loads(zlib.decompress(data).decode("utf-8"))
            if not isinstance(payload, dict):
                continue
            normalized = _normalize_packet(payload, received_at)
            if normalized["schema"] and normalized["schema"] != SPATIAL_STREAM_SCHEMA:
                print(f"[UDP] Warning: unexpected schema {normalized['schema']!r}")
            with state_lock:
                latest_streams[str(normalized["stream_id"])] = normalized
        except Exception as exc:
            print(f"[UDP] Packet parse error: {exc}")

    try:
        sock.close()
    except OSError:
        pass


def render_thread() -> None:
    global latest_render_png, latest_render_error, latest_render_at
    cfg = _config()
    interval = 1.0 / max(0.1, float(cfg.render_hz))
    while not STOP_EVENT.is_set():
        started = time.time()
        snapshot = _build_spatial_map_snapshot()
        png_bytes, error = _render_snapshot(snapshot)
        with render_lock:
            latest_render_png = png_bytes
            latest_render_error = error
            latest_render_at = time.time()
        if png_bytes is not None:
            _write_latest_png(png_bytes)
            _maybe_show_display(png_bytes)
        if error:
            print(f"[Render] {error}")
        elapsed = time.time() - started
        time.sleep(max(0.02, interval - elapsed))


@app.route("/healthz", methods=["GET"])
def healthz():
    with render_lock:
        render_age = None if latest_render_at == 0.0 else time.time() - latest_render_at
    with state_lock:
        stream_count = len(latest_streams)
    with fusion_lock:
        track_count = len(fusion_tracks)
    return jsonify(
        {
            "ok": True,
            "streams_seen": stream_count,
            "fused_tracks": track_count,
            "latest_render_age_s": render_age,
            "latest_render_error": latest_render_error,
        }
    )


@app.route("/api/spatial_map/latest", methods=["GET"])
def get_spatial_map_latest():
    snapshot = _build_spatial_map_snapshot()
    if request.args.get("visualize", "").lower() == "true":
        snapshot["visualization_url"] = "/api/spatial_map/live.png"
        snapshot["visualization_file"] = os.path.join(
            str(_config().output_dir),
            "latest_fusion_object_spatial_map_v2.png",
        )
    return jsonify(snapshot)


@app.route("/api/analytics/counts/latest", methods=["GET"])
def get_counts_latest():
    snapshot = _build_spatial_map_snapshot()
    type_counts = Counter()
    motion_counts = Counter()
    common_counts = Counter()
    for obj in snapshot.get("spatial_map_objects", []):
        type_counts[str(obj.get("type") or "Unknown")] += 1
        motion_state = str(obj.get("motion_state") or "unspecified")
        motion_counts[motion_state] += 1
        common_counts["common" if obj.get("is_common_object") else "single_stream"] += 1
    return jsonify(
        {
            "frame_id": snapshot.get("frame_id"),
            "counts": dict(type_counts),
            "motion_state_counts": dict(motion_counts),
            "association_counts": dict(common_counts),
            "active_streams": snapshot.get("active_streams", []),
        }
    )


@app.route("/api/fusion_streams/latest", methods=["GET"])
def get_fusion_streams_latest():
    snapshot = _build_spatial_map_snapshot()
    return jsonify(
        {
            "active_streams": snapshot.get("active_streams", []),
            "stale_streams": snapshot.get("stale_streams", []),
        }
    )


@app.route("/api/spatial_map/distributions/latest", methods=["GET"])
def get_distributions_latest():
    snapshot = _build_spatial_map_snapshot()
    return jsonify(
        {
            "frame_id": snapshot.get("frame_id"),
            "fused_objects": snapshot.get("spatial_map_objects", []),
            "object_associations": snapshot.get("object_associations", []),
            "raw_spatial_map_objects": snapshot.get("raw_spatial_map_objects", []),
            "fusion_policy": snapshot.get("metadata", {}).get("fusion_policy", {}),
        }
    )


@app.route("/api/spatial_map/live.png", methods=["GET"])
def get_live_png():
    wait_s = min(5.0, max(0.0, _safe_float(request.args.get("wait"), 0.0)))
    deadline = time.time() + wait_s
    while True:
        with render_lock:
            png_bytes = latest_render_png
            render_error = latest_render_error
        if png_bytes is not None:
            return Response(png_bytes, mimetype="image/png")
        if time.time() >= deadline:
            return jsonify({"error": render_error or "No rendered spatial map is available yet"}), 503
        time.sleep(0.05)


@app.route("/api/spatial_map/viewer", methods=["GET"])
def get_live_viewer():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-Client Fused Object Spatial Map</title>
  <style>
    html, body { margin: 0; height: 100%; background: #080b10; }
    img { width: 100vw; height: 100vh; object-fit: contain; display: block; }
  </style>
</head>
<body>
  <img id="map" alt="Multi-client fused object spatial map">
  <script>
    const img = document.getElementById("map");
    function refresh() {
      img.src = "/api/spatial_map/live.png?wait=1&t=" + Date.now();
    }
    img.onload = () => setTimeout(refresh, 250);
    img.onerror = () => setTimeout(refresh, 500);
    refresh();
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


def main() -> None:
    global CONFIG
    CONFIG = parse_args()
    os.makedirs(str(CONFIG.output_dir), exist_ok=True)

    udp_thread = threading.Thread(target=udp_listener_thread, daemon=True)
    map_render_thread = threading.Thread(target=render_thread, daemon=True)
    udp_thread.start()
    map_render_thread.start()

    print(f"Starting multi-client fused spatial map API on {CONFIG.api_host}:{CONFIG.api_port}")
    print(f"Viewer: http://127.0.0.1:{CONFIG.api_port}/api/spatial_map/viewer")
    try:
        app.run(host=str(CONFIG.api_host), port=int(CONFIG.api_port), threaded=True)
    finally:
        STOP_EVENT.set()


if __name__ == "__main__":
    main()

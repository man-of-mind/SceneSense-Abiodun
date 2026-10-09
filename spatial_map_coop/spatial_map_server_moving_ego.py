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
import sys
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

try:
    from spatial_map_coop.multi_ue_v1 import (
        AssociationPolicy,
        MultiUEContractError,
        MultiUESpatialMapService,
    )
    from spatial_map_coop.coop_input_v1 import live_metrics_v1, route_contract_v1
    from spatial_map_coop.coop_input_v1.normalized import CoopInputError
except ModuleNotFoundError:
    # Direct-script invocation from neu_collab/ places this file's directory,
    # rather than abiodun/, on sys.path.
    from multi_ue_v1 import AssociationPolicy, MultiUEContractError, MultiUESpatialMapService
    from coop_input_v1 import live_metrics_v1, route_contract_v1
    from coop_input_v1.normalized import CoopInputError

# This server lives in abiodun/spatial_map_coop/ but reuses extract_traffic_lights.py
# from the neu_collab root; make that importable regardless of CWD.
_NEU_COLLAB_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
if _NEU_COLLAB_ROOT not in sys.path:
    sys.path.insert(0, _NEU_COLLAB_ROOT)

import extract_traffic_lights as traffic_map


app = Flask(__name__)

DEFAULT_UDP_PORT = 39201
DEFAULT_API_PORT = 35011
SPATIAL_STREAM_SCHEMA = "fusion_object_spatial_map.v1"
FUSED_SPATIAL_MAP_SCHEMA = "fusion_object_spatial_map.fused.v2"
SPLITFUSION_EDGE_RESULT_SCHEMA = "splitfusion_edge_result.v2"
MULTI_UE_LIVE_VIEW_UPDATE_SCHEMA = "scenesense.multi_ue_live_view_update.v1"
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
multi_ue_service: Optional[MultiUESpatialMapService] = None
multi_ue_source_map: Dict[str, str] = {}
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

# Presentation/metrics state. None of it participates in association.
VIEWER_ASSET_DIR = os.path.join(SCRIPT_DIR, "viewer_assets")
metrics_registry = live_metrics_v1.MetricsRegistry()
route_contract: Optional["route_contract_v1.RouteContract"] = None
route_contract_error: Optional[str] = None
ingest_counters: Counter = Counter()
ingest_counters_lock = threading.Lock()
# Last multi-UE service diagnostic, captured at the one already-authorized
# stateful read (_build_multi_ue_v1_objects). Health and metrics endpoints read
# this cache instead of calling snapshot_latest() themselves: snapshot_latest()
# advances the conservative track store, so calling it from a status endpoint
# would let a monitoring poll mutate scientific association state.
service_diagnostic_lock = threading.Lock()
latest_service_diagnostic: Dict[str, object] = {
    "status": "WAITING_FOR_REGISTERED_UE_UPDATES"
}


def _read_viewer_asset(name: str) -> str:
    """Read one viewer asset from disk. Cached per process, never per request."""

    path = os.path.join(VIEWER_ASSET_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _viewer_html() -> str:
    """Inline the presentation assets into the viewer shell.

    The served bytes are the asset files verbatim, so the JavaScript exercised by
    the offline node tests is exactly the JavaScript the browser receives.
    """

    html = _read_viewer_asset("viewer_v2.html")
    for asset in (
        "presentation_predictor_v1.js",
        "presentation_inventory_v1.js",
        "viewer_app_v2.js",
    ):
        marker = f"/*INLINE:{asset}*/"
        if marker not in html:
            raise RuntimeError(f"viewer shell lacks the {asset} inline marker")
        html = html.replace(marker, _read_viewer_asset(asset))
    return html


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
    parser.add_argument(
        "--focus-follow-stream-id",
        default="",
        help=(
            "Center the top-down ROI on the LIVE pose of this stream's anchor (e.g. a moving "
            "ego vehicle), following it as it drives. Takes precedence over --focus-traffic-light-ids. "
            "The ROI is a --focus-radius-m box around the ego's current position."
        ),
    )
    parser.add_argument(
        "--focus-follow-forward-bias",
        type=float,
        default=0.0,
        help=(
            "Shift the follow ROI this fraction of --focus-radius-m forward along the ego heading "
            "(0 = centered on ego; 0.5 = biased ahead, showing more road in front than behind)."
        ),
    )
    parser.add_argument(
        "--multi-ue-v1",
        action="store_true",
        help=(
            "Use the conservative multi-UE v1 filtering/Hungarian/selected-source "
            "track path instead of the historical greedy confidence-weighted fusion."
        ),
    )
    parser.add_argument(
        "--multi-ue-source",
        action="append",
        default=[],
        metavar="STREAM_ID=UE_ID",
        help="Explicit stable stream-to-UE identity; repeat once per live UE.",
    )
    parser.add_argument(
        "--multi-ue-session-id",
        default="",
        help="Required stable drive/session identity when --multi-ue-v1 is enabled.",
    )
    parser.add_argument(
        "--multi-ue-maximum-age-ms",
        type=float,
        default=500.0,
        help="Initial engineering freshness limit; report separately from service readiness.",
    )
    parser.add_argument(
        "--multi-ue-alignment-ms",
        type=float,
        default=100.0,
        help="Maximum source-frame capture spread admitted to association.",
    )
    parser.add_argument(
        "--route-mode",
        choices=route_contract_v1.ROUTE_MODES,
        default=route_contract_v1.ROUTE_MODE_STAGE2,
        help=(
            "Route contract for the presentation overlay. 'stage2' is the "
            "historical configurable spawn-index loop and is NOT Route B. "
            "'route-b-passive' binds and hash-verifies the canonical "
            "Town10HD_Opt Route-B loop; it does not authorize a Route-B run."
        ),
    )
    parser.add_argument(
        "--route-overlay",
        action="store_true",
        help=(
            "Draw the canonical Route-B reference path as a background overlay. "
            "Requires --route-mode route-b-passive and a hash-verified route."
        ),
    )
    parser.add_argument(
        "--route-overlay-stride",
        type=int,
        default=4,
        help="Subsample every Nth canonical route waypoint for the display overlay.",
    )
    parser.add_argument(
        "--metrics-window",
        type=int,
        default=live_metrics_v1.DEFAULT_WINDOW,
        help="Bounded rolling window length for the live latency panel.",
    )
    return parser.parse_args()


def _parse_multi_ue_sources(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    used_ue_ids: Set[str] = set()
    for raw_value in values:
        stream_id, separator, ue_id = str(raw_value).partition("=")
        stream_id = stream_id.strip()
        ue_id = ue_id.strip()
        if not separator or not stream_id or not ue_id:
            raise ValueError("--multi-ue-source must use nonempty STREAM_ID=UE_ID")
        if stream_id in result:
            raise ValueError(f"duplicate multi-UE stream identity: {stream_id}")
        if ue_id in used_ue_ids:
            raise ValueError(f"duplicate stable UE identity: {ue_id}")
        result[stream_id] = ue_id
        used_ue_ids.add(ue_id)
    return result


def _initialize_multi_ue_service() -> None:
    global multi_ue_service, multi_ue_source_map
    cfg = _config()
    if not bool(cfg.multi_ue_v1):
        multi_ue_service = None
        multi_ue_source_map = {}
        return
    session_id = str(cfg.multi_ue_session_id or "").strip()
    if not session_id:
        raise ValueError("--multi-ue-session-id is required with --multi-ue-v1")
    multi_ue_source_map = _parse_multi_ue_sources(cfg.multi_ue_source)
    if len(multi_ue_source_map) < 2:
        raise ValueError("--multi-ue-v1 requires at least two explicit UE sources")
    maximum_age_ns = round(float(cfg.multi_ue_maximum_age_ms) * 1_000_000)
    alignment_ns = round(float(cfg.multi_ue_alignment_ms) * 1_000_000)
    policy = AssociationPolicy(
        maximum_observation_age_ns=maximum_age_ns,
        alignment_tolerance_ns=alignment_ns,
        maximum_pair_time_delta_ns=alignment_ns,
        maximum_xy_distance_m=float(cfg.association_radius_m),
        maximum_relative_size_difference=float(cfg.association_dimension_ratio),
        track_match_distance_m=float(cfg.track_match_radius_m),
        track_stale_after_ns=round(float(cfg.track_stale_s) * 1_000_000_000),
        minimum_confidence=float(cfg.min_object_score),
        confidence_provenance=(
            "UPSTREAM_SOURCE_FILTERED_OUTPUT_NO_ADDITIONAL_SCORE_GATE"
            if float(cfg.min_object_score) == 0.0
            else "OPERATOR_CONFIGURED_DEMO_SCORE_GATE_NOT_CALIBRATED_UNCERTAINTY"
        ),
    )
    multi_ue_service = MultiUESpatialMapService(policy)


def _initialize_presentation_contracts() -> None:
    """Bind the route contract and size the metrics window.

    A route-contract failure is reported and disables the overlay; it never
    silently degrades into an unverified overlay, and it never blocks map
    ingestion.
    """

    global route_contract, route_contract_error, metrics_registry
    cfg = _config()
    metrics_registry = live_metrics_v1.MetricsRegistry(
        capacity=max(1, int(cfg.metrics_window))
    )
    stream_ids = tuple(sorted(multi_ue_source_map))
    ue_ids = tuple(multi_ue_source_map[stream_id] for stream_id in stream_ids)
    try:
        route_contract = route_contract_v1.build_route_contract(
            mode=str(cfg.route_mode),
            stream_ids=stream_ids or ("unbound",),
            ue_ids=ue_ids or ("unbound",),
        )
        route_contract_error = None
    except CoopInputError as exc:
        route_contract = None
        route_contract_error = str(exc)
        print(f"[Route] contract not bound: {exc}")
    if bool(cfg.route_overlay):
        if route_contract is None or not route_contract.is_route_b:
            route_contract_error = (
                route_contract_error
                or "--route-overlay requires --route-mode route-b-passive"
            )
            print(f"[Route] overlay disabled: {route_contract_error}")


def _route_overlay_payload() -> Optional[Dict[str, object]]:
    """Verified canonical route points for the display overlay, or None."""

    cfg = _config()
    if not bool(cfg.route_overlay) or route_contract is None:
        return None
    if not route_contract.is_route_b or route_contract.route is None:
        return None
    points = route_contract.overlay_polyline(stride=max(1, int(cfg.route_overlay_stride)))
    if len(points) < 2:
        return None
    return {
        "points": [[float(x), float(y)] for x, y in points],
        "route_name": route_contract.route.name,
        "route_sha256": route_contract.route.sha256,
        "presentation_label": route_contract.presentation_label,
        "qualification_status": route_contract.qualification_status,
        "note": "display overlay only; it never affects measured XY or association",
    }


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


def _required_finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MultiUEContractError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise MultiUEContractError(f"{label} must be finite")
    return result


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


def _follow_view_bounds(streams: Sequence[Dict[str, object]], now: float) -> Optional[Dict[str, object]]:
    """ROI that tracks a moving ego: a --focus-radius-m box around the live anchor pose of the
    stream named by --focus-follow-stream-id. Returns None if not configured or the stream is
    absent/stale so the caller can fall back to the static traffic-light focus."""
    cfg = _config()
    follow_id = str(getattr(cfg, "focus_follow_stream_id", "") or "").strip()
    if not follow_id:
        return None

    match = None
    for stream in streams:
        if str(stream.get("stream_id")) != follow_id:
            continue
        if now - float(stream.get("received_at", 0.0)) > float(cfg.stream_stale_s):
            continue
        match = stream
        break

    if match is None:
        return {
            "mode": "follow_stream",
            "follow_stream_id": follow_id,
            "bounds": None,
            "warning": f"Follow stream '{follow_id}' not seen recently; showing full map.",
        }

    # The moving-ego client ships its live sensor pose in the packet's "camera" block
    # (location + rotation); fall back to an "anchor.transform" if a future client sends one.
    cam = match.get("camera") or {}
    transform = (match.get("anchor") or {}).get("transform") or {}
    loc = cam.get("location") or transform.get("location") or {}
    rot = cam.get("rotation") or transform.get("rotation") or {}
    cx = _safe_float(loc.get("x"))
    cy = _safe_float(loc.get("y"))
    yaw_deg = _safe_float(rot.get("yaw"))

    radius_m = max(1.0, float(cfg.focus_radius_m))
    padding_m = max(0.0, float(cfg.focus_padding_m))
    bias = float(getattr(cfg, "focus_follow_forward_bias", 0.0) or 0.0)
    if bias:  # push the box ahead of the car along its heading (CARLA: +x at yaw=0)
        yaw_rad = math.radians(yaw_deg)
        cx += bias * radius_m * math.cos(yaw_rad)
        cy += bias * radius_m * math.sin(yaw_rad)

    return {
        "mode": "follow_stream",
        "follow_stream_id": follow_id,
        "radius_m": radius_m,
        "padding_m": padding_m,
        "forward_bias": bias,
        "ego_pose": {"x": cx, "y": cy, "yaw_deg": yaw_deg},
        "bounds": {
            "x_min": float(cx - radius_m - padding_m),
            "x_max": float(cx + radius_m + padding_m),
            "y_min": float(cy - radius_m - padding_m),
            "y_max": float(cy + radius_m + padding_m),
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
        # The town geometry is static within a run. Once real roads/buildings are loaded, NEVER re-query
        # CARLA — the periodic re-query (default every 10s) is a 1-2s blocking round-trip on the request
        # path that froze the live map. Only the throttled retry path below runs if the first build failed
        # (empty geometry), so a transient CARLA hiccup can still recover.
        if cache_ready and (static_map_cache.get("roads") or static_map_cache.get("buildings")):
            return dict(static_map_cache)
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
    normalized_type = object_type.strip().lower().replace("_", "")
    motion_state = str(obj.get("motion_state") or "")
    if normalized_type == "parkedvehicle":
        object_type = "Vehicle"
        motion_state = motion_state or "parked"
    elif normalized_type == "movingvehicle":
        object_type = "Vehicle"
        motion_state = motion_state or "moving"
    elif normalized_type in ("vehicle", "car", "truck", "bus"):
        object_type = "Vehicle"
    elif normalized_type in ("person", "pedestrian", "walker"):
        # model class name is "person"; map to the canonical map label so it gets the
        # Pedestrian color/legend instead of falling through to white "Unknown".
        object_type = "Pedestrian"
    elif normalized_type in ("cyclist", "bicycle", "bike"):
        object_type = "Cyclist"

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
        "anchor": payload.get("anchor") if isinstance(payload.get("anchor"), dict) else {},
        "camera": payload.get("camera") if isinstance(payload.get("camera"), dict) else {},
        "segmentation": payload.get("segmentation") if isinstance(payload.get("segmentation"), dict) else {},
        "latency": payload.get("latency") if isinstance(payload.get("latency"), dict) else {},
        "objects": objects,
        "object_count": len(objects),
        "source_script": str(payload.get("source_script") or ""),
    }


def _ingest_splitfusion_live_view_update(
    envelope: Dict[str, object], received_at: float
) -> Dict[str, object]:
    """Ingest one compact action-profile result and expose its raw reports to the viewer."""

    if multi_ue_service is None:
        raise MultiUEContractError("multi-UE v1 is not enabled")
    if envelope.get("schema") != MULTI_UE_LIVE_VIEW_UPDATE_SCHEMA:
        raise MultiUEContractError("unexpected live-view update schema")
    ue_id = str(envelope.get("ue_id") or "").strip()
    session_id = str(envelope.get("session_id") or "").strip()
    if not ue_id or session_id != str(_config().multi_ue_session_id):
        raise MultiUEContractError("live-view UE/session identity mismatch")
    if envelope.get("clock_domain") != "carla_simulation_ns":
        raise MultiUEContractError("live-view update must use carla_simulation_ns")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("schema") != SPLITFUSION_EDGE_RESULT_SCHEMA:
        raise MultiUEContractError("live-view update lacks a SplitFusion edge result")
    stream_id = str(payload.get("stream_id") or "")
    if multi_ue_source_map.get(stream_id) != ue_id:
        raise MultiUEContractError("live-view stream is not registered to this UE")
    ego_pose = envelope.get("ego_pose")
    if not isinstance(ego_pose, dict):
        raise MultiUEContractError("live-view update lacks an ego pose")
    location = ego_pose.get("location")
    rotation = ego_pose.get("rotation")
    if not isinstance(location, dict) or not isinstance(rotation, dict):
        raise MultiUEContractError("live-view ego pose is malformed")
    normalized_location = {
        axis: _required_finite(location.get(axis), f"ego_pose.location.{axis}")
        for axis in ("x", "y", "z")
    }
    normalized_rotation = {
        axis: _required_finite(rotation.get(axis), f"ego_pose.rotation.{axis}")
        for axis in ("pitch", "yaw", "roll")
    }
    capture_timestamp_ns = _safe_int(payload.get("capture_timestamp_ns"), -1)
    if capture_timestamp_ns < 0:
        raise MultiUEContractError("capture_timestamp_ns must be nonnegative")
    result = multi_ue_service.ingest_splitfusion(
        payload,
        ue_id=ue_id,
        session_id=session_id,
        received_timestamp_ns=capture_timestamp_ns,
        clock_domain="carla_simulation_ns",
    )
    update = payload.get("object_map_update")
    records = update.get("records") if isinstance(update, dict) else None
    if not isinstance(records, list):
        raise MultiUEContractError("SplitFusion object records must be a list")
    frame_id = _safe_int(payload.get("frame_id"), -1)
    objects = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MultiUEContractError("SplitFusion object record must be an object")
        viewer_record = dict(record)
        viewer_record["id"] = str(
            record.get("candidate_identity", record.get("id", f"observation_{index}"))
        )
        viewer_record["type"] = str(record.get("class_name", record.get("type", "Unknown")))
        normalized = _normalize_object(
            viewer_record, stream_id=stream_id, frame_id=frame_id, index=index
        )
        if normalized is not None:
            objects.append(normalized)
    stream = {
        "schema": SPLITFUSION_EDGE_RESULT_SCHEMA,
        "stream_id": stream_id,
        "node_id": ue_id,
        "traffic_light_id": "",
        "traffic_light_actor_id": -1,
        "traffic_light_opendrive_id": "",
        "frame_id": frame_id,
        "timestamp": received_at,
        "carla_timestamp": capture_timestamp_ns / 1_000_000_000.0,
        # The measured capture stamp and the NAME of its clock travel together.
        # This is the CARLA simulation clock; it must never be subtracted from
        # received_at, which is a unix wall clock.
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "capture_clock_domain": "carla_simulation_ns",
        "received_at": received_at,
        "anchor": {
            "actor_id": _safe_int(ego_pose.get("actor_id"), -1),
            "transform": {
                "location": normalized_location,
                "rotation": normalized_rotation,
            },
        },
        "camera": {
            "location": normalized_location,
            "rotation": normalized_rotation,
            "fov": _required_finite(envelope.get("horizontal_fov_deg", 120.0), "horizontal_fov_deg"),
        },
        "segmentation": {},
        "latency": {},
        "objects": objects,
        "object_count": len(objects),
        "source_script": "live_two_ue_action50_v1",
        "action_id": _safe_int(payload.get("action_id"), -1),
        "profile_id": str(payload.get("profile_id") or ""),
    }
    with state_lock:
        latest_streams[stream_id] = stream
    # Presentation metrics only: a bounded counter bump and one rate sample.
    # Neither touches association state, and neither can block ingestion.
    _bump_ingest_counter(result.disposition)
    metrics_registry.observe_receipt(
        stream_id, frame_id=frame_id, received_wall_s=received_at
    )
    return {
        "status": "ACCEPTED",
        "disposition": result.disposition,
        "ue_id": ue_id,
        "stream_id": stream_id,
        "frame_id": frame_id,
        "object_count": len(objects),
    }


def _bump_ingest_counter(disposition: str) -> None:
    key = {
        "ACCEPTED": "accepted",
        "DUPLICATE_IDENTICAL": "duplicate_identical",
    }.get(str(disposition), str(disposition).lower())
    with ingest_counters_lock:
        ingest_counters[key] += 1


def _ingest_counter_snapshot() -> Dict[str, int]:
    with ingest_counters_lock:
        counters = dict(ingest_counters)
    # Declare the registered keys explicitly so the panel can show 0 for a
    # counter that genuinely is zero and N/A for one that does not exist.
    for key in ("accepted", "duplicate_identical", "conflict", "rejected"):
        counters.setdefault(key, 0)
    return counters


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


def _build_multi_ue_v1_objects() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Expose conservative tracks using the historical viewer's object shape."""
    if multi_ue_service is None:
        return [], [], {}
    service_snapshot = multi_ue_service.snapshot_latest(clock_domain="carla_simulation_ns")
    if service_snapshot is None:
        return [], [], {"status": "WAITING_FOR_REGISTERED_UE_UPDATES"}
    document = service_snapshot.as_dict()
    current_association_ids = {
        str(association["association_id"])
        for association in document["associations"]
    }
    retained_track_count = len(document["tracks"])
    tracks = []
    for track in document["tracks"]:
        # The service deliberately retains tracks across missed observations. The
        # raw/associated presentation comparison must instead use the same current
        # observation snapshot on both sides; otherwise retained history can make
        # "association on" appear to create objects. Persistence stays in service state.
        if str(track["association_id"]) not in current_association_ids:
            continue
        source_keys = track["contributing_sources"]
        tracks.append(
            {
                "track_id": track["track_id"],
                "type": track["class_name"],
                "score": track["score"],
                "support_stream_count": len(source_keys),
                "source_stream_ids": sorted({source[2] for source in source_keys}),
                "is_common_object": len(source_keys) >= max(1, int(_config().common_min_streams)),
                "location": {
                    "x": track["world_xyz"][0],
                    "y": track["world_xyz"][1],
                    "z": track["world_xyz"][2],
                },
                "dimensions": {
                    "length": track["size_lwh"][0],
                    "width": track["size_lwh"][1],
                    "height": track["size_lwh"][2],
                },
                "yaw_deg": _object_map_yaw_deg(track["yaw_deg"]),
                "model_yaw_deg": track["yaw_deg"],
                "selected_ue_id": track["selected_ue_id"],
                "selected_session_id": track["selected_session_id"],
                "selected_stream_id": track["selected_stream_id"],
                "source_stream_id": track["selected_stream_id"],
                "selected_frame_id": track["selected_frame_id"],
                "selected_observation_id": track["selected_observation_id"],
                "selected_raw_record_sha256": track["selected_raw_record_sha256"],
                "capture_timestamp_ns": track["capture_timestamp_ns"],
                "age_ns": track["age_ns"],
                "contributing_observation_hashes": track[
                    "contributing_observation_hashes"
                ],
                "association_id": track["association_id"],
                "update_count": track["update_count"],
                "fusion_method": "SELECTED_FRESHEST_SOURCE_NO_POSITION_AVERAGING",
                "distribution": {
                    "source_stream_count": len(source_keys),
                    "source_stream_ids": sorted({source[2] for source in source_keys}),
                    "position_covariance_available": False,
                    "confidence_used_as_position_covariance": False,
                },
            }
        )
    document["viewer_projection"] = {
        "policy": "CURRENT_ASSOCIATION_TRACKS_ONLY",
        "rendered_track_count": len(tracks),
        "retained_service_track_count": retained_track_count,
        "historical_tracks_not_rendered": retained_track_count - len(tracks),
    }
    _record_service_diagnostic(document, rendered_track_count=len(tracks))
    return tracks, document["associations"], document


def _record_service_diagnostic(document: Dict[str, object], *, rendered_track_count: int) -> None:
    """Cache a read-only projection of the service snapshot for status endpoints."""

    rejected = Counter(
        str(item.get("reason")) for item in (document.get("rejected_observations") or [])
    )
    associations = document.get("associations") or []
    tracks = document.get("tracks") or []
    diagnostic: Dict[str, object] = {
        "status": "ok",
        "schema": document.get("schema"),
        "clock_domain": document.get("clock_domain"),
        "snapshot_timestamp_ns": document.get("snapshot_timestamp_ns"),
        "agent_binding_status": document.get("agent_binding_status"),
        "input_source_count": document.get("input_source_count"),
        "aligned_source_count": document.get("aligned_source_count"),
        "accepted_observation_count": document.get("accepted_observation_count"),
        "rejected_observation_count": len(document.get("rejected_observations") or []),
        "rejection_reasons": dict(sorted(rejected.items())),
        "association_count": len(associations),
        "multi_source_association_count": sum(
            1 for item in associations if int(item.get("source_count", 0)) >= 2
        ),
        "retained_service_track_count": len(tracks),
        "current_association_track_count": int(rendered_track_count),
        "historical_tracks_not_rendered": len(tracks) - int(rendered_track_count),
        "viewer_projection": document.get("viewer_projection"),
    }
    with service_diagnostic_lock:
        latest_service_diagnostic.clear()
        latest_service_diagnostic.update(diagnostic)


def _stream_capture_timestamp_ns(stream: Dict[str, object]) -> Optional[int]:
    """The stream's measured capture stamp in ns, or None when it has none."""

    explicit = stream.get("capture_timestamp_ns")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return int(explicit)
    carla_timestamp = _safe_float(stream.get("carla_timestamp"), 0.0)
    if carla_timestamp > 0.0 and math.isfinite(carla_timestamp):
        return round(carla_timestamp * 1_000_000_000)
    return None


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
        cam = stream.get("camera") or {}
        cam_loc = cam.get("location") or {}
        cam_rot = cam.get("rotation") or {}
        stream_info = {
            "stream_id": stream["stream_id"],
            "node_id": stream["node_id"],
            "traffic_light_id": stream["traffic_light_id"],
            "traffic_light_actor_id": stream["traffic_light_actor_id"],
            "traffic_light_opendrive_id": stream["traffic_light_opendrive_id"],
            "anchor": stream.get("anchor", {}),
            # per-stream sensor pose + FoV (from the packet's camera block) — enables cooperative
            # per-stream FoV polygons + occlusion reasoning on real data (not just the followed ego).
            "sensor_pose": {"x": _safe_float(cam_loc.get("x")), "y": _safe_float(cam_loc.get("y")),
                            "yaw_deg": _safe_float(cam_rot.get("yaw"))},
            "fov_deg": _safe_float(cam.get("fov"), 90.0),
            "frame_id": stream["frame_id"],
            "age_s": age_s,
            # Measured capture stamp for this stream, in its own declared clock
            # domain. The viewer's causal display predictor derives velocity from
            # this (a real measured clock) while bounding its extrapolation
            # horizon in local browser time. Absent -> None, never zero, so a
            # legacy packet without a capture clock is position-stepped instead
            # of being given a fabricated velocity.
            "capture_timestamp_ns": _stream_capture_timestamp_ns(stream),
            "capture_clock_domain": str(stream.get("capture_clock_domain") or ""),
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
    multi_ue_diagnostic: Dict[str, object] = {}
    if multi_ue_service is None:
        fused_objects, object_associations = _fuse_and_smooth_objects(raw_objects)
    elif not active_streams:
        fused_objects, object_associations = [], []
        multi_ue_diagnostic = {"status": "NO_WALL_CLOCK_FRESH_REGISTERED_STREAM"}
    else:
        fused_objects, object_associations, multi_ue_diagnostic = _build_multi_ue_v1_objects()
        if bool(cfg.hide_single_stream_objects):
            fused_objects = [obj for obj in fused_objects if bool(obj["is_common_object"])]
        fused_objects = fused_objects[: max(0, int(cfg.max_rendered_objects))]
    active_keys = _active_anchor_keys(streams, now)
    focus_ids = _focus_traffic_light_ids()
    anchors = []
    for anchor in static_context.get("traffic_lights") or []:
        anchor_id = str(anchor.get("id"))
        opendrive_id = str(anchor.get("opendrive_id") or "")
        active = anchor_id in active_keys or (opendrive_id and opendrive_id in active_keys)
        focused = _anchor_matches_focus(anchor, focus_ids)
        anchors.append({**anchor, "active": bool(active), "focus": bool(focused)})
    # A live follow-ego ROI (if configured) takes precedence over the static traffic-light focus.
    focus_view = _follow_view_bounds(streams, now) or _focus_view_bounds(anchors)

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
                "implementation": (
                    "CONSERVATIVE_MULTI_UE_V1"
                    if multi_ue_service is not None
                    else "HISTORICAL_GREEDY_CONFIDENCE_WEIGHTED"
                ),
                "association_radius_m": float(cfg.association_radius_m),
                "association_dimension_ratio": float(cfg.association_dimension_ratio),
                "common_min_streams": int(cfg.common_min_streams),
                "track_match_radius_m": float(cfg.track_match_radius_m),
                "smoothing_alpha": float(cfg.smoothing_alpha),
                "hide_single_stream_objects": bool(cfg.hide_single_stream_objects),
                "object_yaw_map_offset_deg": float(cfg.object_yaw_map_offset_deg),
            },
            "multi_ue_v1": multi_ue_diagnostic,
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
            if multi_ue_service is not None:
                stream_id = str(normalized["stream_id"])
                ue_id = multi_ue_source_map.get(stream_id)
                if ue_id is None:
                    raise MultiUEContractError(
                        f"unregistered multi-UE stream identity: {stream_id}"
                    )
                multi_ue_service.ingest_legacy(
                    payload,
                    ue_id=ue_id,
                    session_id=str(cfg.multi_ue_session_id),
                    received_timestamp_ns=round(received_at * 1_000_000_000),
                )
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


def _multi_ue_service_state() -> Dict[str, object]:
    """Report the actual MultiUESpatialMapService state.

    This replaces the previous reliance on ``fusion_tracks``, which belongs to
    the historical greedy confidence-weighted path. When ``--multi-ue-v1`` is
    enabled that dictionary is never populated, so a health check reading it
    reported zero tracks for a perfectly healthy conservative service.
    """

    if multi_ue_service is None:
        with fusion_lock:
            historical_tracks = len(fusion_tracks)
        return {
            "implementation": "HISTORICAL_GREEDY_CONFIDENCE_WEIGHTED",
            "multi_ue_v1_enabled": False,
            "historical_fusion_tracks": historical_tracks,
        }

    state: Dict[str, object] = {
        "implementation": "CONSERVATIVE_MULTI_UE_V1",
        "multi_ue_v1_enabled": True,
        "registered_sources": dict(sorted(multi_ue_source_map.items())),
        "buffered_source_count": int(multi_ue_service.buffer.source_count()),
        "policy": multi_ue_service.policy.as_dict(),
    }
    with service_diagnostic_lock:
        state.update(dict(latest_service_diagnostic))
    return state


@app.route("/healthz", methods=["GET"])
def healthz():
    with render_lock:
        render_age = None if latest_render_at == 0.0 else time.time() - latest_render_at
    with state_lock:
        stream_count = len(latest_streams)
    service_state = _multi_ue_service_state()
    return jsonify(
        {
            "ok": True,
            "streams_seen": stream_count,
            # Track counts now come from the service that is actually running.
            "map_tracks": service_state.get(
                "current_association_track_count",
                service_state.get("historical_fusion_tracks", 0),
            ),
            "multi_ue_service": service_state,
            "ingest_counters": _ingest_counter_snapshot(),
            "route_contract": (
                None if route_contract is None else route_contract.as_dict()
            ),
            "route_contract_error": route_contract_error,
            "latest_render_age_s": render_age,
            "latest_render_error": latest_render_error,
        }
    )


@app.route("/api/multi_ue/v1/live_updates/healthz", methods=["GET"])
def multi_ue_live_updates_healthz():
    return jsonify(
        {
            "status": "ok" if multi_ue_service is not None else "disabled",
            "schema": MULTI_UE_LIVE_VIEW_UPDATE_SCHEMA,
            "session_id": str(getattr(_config(), "multi_ue_session_id", "")),
            "registered_sources": dict(sorted(multi_ue_source_map.items())),
            # Real service state, not the historical fusion_tracks structure.
            "multi_ue_service": _multi_ue_service_state(),
            "ingest_counters": _ingest_counter_snapshot(),
        }
    ), (200 if multi_ue_service is not None else 503)


@app.route("/api/multi_ue/v1/live_updates", methods=["POST"])
def post_multi_ue_live_update():
    try:
        envelope = request.get_json(force=True)
        if not isinstance(envelope, dict):
            raise MultiUEContractError("live-view update must be an object")
        started = time.perf_counter()
        result = _ingest_splitfusion_live_view_update(envelope, time.time())
        metrics_registry.observe_demo_post_ms((time.perf_counter() - started) * 1000.0)
        return jsonify(result), 202
    except (MultiUEContractError, TypeError, ValueError) as exc:
        message = str(exc)
        _bump_ingest_counter(
            "conflict" if "conflict" in message or "identity" in message else "rejected"
        )
        return jsonify({"error": message}), 422


@app.route("/api/spatial_map/latest", methods=["GET"])
def get_spatial_map_latest():
    started = time.perf_counter()
    snapshot = _build_spatial_map_snapshot()
    metrics_registry.observe_snapshot_build_ms((time.perf_counter() - started) * 1000.0)
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


@app.route("/api/spatial_map/static_geometry", methods=["GET"])
def get_static_geometry():
    """Static town backdrop (roads + building footprints) for the canvas viewer. Fetched once by
    the viewer; the underlying geometry is cached in _refresh_static_map_context()."""
    ctx = _refresh_static_map_context()

    def _xy(p):
        if isinstance(p, dict):
            return [_safe_float(p.get("x")), _safe_float(p.get("y"))]
        return [_safe_float(p[0]), _safe_float(p[1])]

    roads = []
    for polyline in (ctx.get("roads") or []):
        pts = [_xy(p) for p in polyline]
        if len(pts) >= 2:
            roads.append(pts)
    buildings = []
    for building in (ctx.get("buildings") or []):
        footprint = building.get("footprint") if isinstance(building, dict) else None
        if not footprint:
            continue
        pts = [_xy(p) for p in footprint]
        if len(pts) >= 3:
            buildings.append(pts)
    return jsonify(
        {
            "map_name": ctx.get("map_name"),
            "error": ctx.get("error"),
            "roads": roads,
            "buildings": buildings,
            # Present only when a hash-verified canonical Route-B contract is
            # bound and the operator asked for the overlay.
            "route_overlay": _route_overlay_payload(),
            "route_overlay_error": route_contract_error,
        }
    )


@app.route("/api/spatial_map/metrics", methods=["GET"])
def get_live_metrics():
    """Rolling live metrics for the viewer panel.

    Read-only. It reports bounded rolling windows, the cached service
    diagnostic, ingest counters and the transport's honest labels. It never
    computes a cross-clock age as a number and never substitutes zero for an
    absent field.
    """

    now = time.time()
    report = metrics_registry.report(now_wall_s=now)
    with state_lock:
        stale_streams = sum(
            1
            for stream in latest_streams.values()
            if now - float(stream["received_at"]) > float(_config().stream_stale_s)
        )
    counters = _ingest_counter_snapshot()
    counters["stale_streams"] = stale_streams
    service_state = _multi_ue_service_state()

    # Transport honesty block. The demonstration path is HTTP on localhost with a
    # CARLA-simulation capture clock, so its wall AoI is not computable and its
    # POST duration is not a radio measurement.
    if multi_ue_service is not None:
        demo = live_metrics_v1.demo_transport_labels()
        transport = {
            "label": live_metrics_v1_transport_label(),
            "capture_clock_domain": "carla_simulation_ns",
            "receipt_clock_domain": "unix_wall_ns",
            "cross_clock_wall_aoi": demo["cross_clock_wall_aoi"],
            "http_post_duration": demo["http_post_duration"],
            "model_execution": demo["model_execution"],
            "radio_claim": demo["radio_claim"],
        }
    else:
        transport = {
            "label": "LEGACY_UDP_POLE_CLIENT_DEMO",
            "capture_clock_domain": "carla_simulation_ns",
            "receipt_clock_domain": "unix_wall_ns",
            "cross_clock_wall_aoi": live_metrics_v1.INCOMPATIBLE_CLOCKS,
            "radio_claim": "none - this transport carries no radio",
        }

    report.update(
        {
            "ingest_counters": counters,
            "multi_ue_service": service_state,
            "transport": transport,
            "route_contract": (
                None if route_contract is None else route_contract.as_dict()
            ),
            "route_contract_error": route_contract_error,
            "direct_edge_timing_available": False,
            "direct_edge_timing_note": (
                "this transport carries no edge stage stamps or map terminal; "
                "direct edge-to-map latency charts stay empty by contract"
            ),
        }
    )
    return jsonify(report)


def live_metrics_v1_transport_label() -> str:
    return live_metrics_v1.NOT_AVAILABLE if multi_ue_service is None else (
        "HTTP_JSON_POST_LOCALHOST_DEMO"
    )


# Legacy matplotlib-PNG viewer kept as a fallback.
@app.route("/api/spatial_map/viewer_png", methods=["GET"])
def get_live_viewer_png():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-Client Fused Object Spatial Map (PNG)</title>
  <style>
    html, body { margin: 0; height: 100%; background: #080b10; }
    img { width: 100vw; height: 100vh; object-fit: contain; display: block; }
  </style>
</head>
<body>
  <img id="map" alt="Multi-client fused object spatial map">
  <script>
    const img = document.getElementById("map");
    function refresh() { img.src = "/api/spatial_map/live.png?wait=1&t=" + Date.now(); }
    img.onload = () => setTimeout(refresh, 250);
    img.onerror = () => setTimeout(refresh, 500);
    refresh();
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


# Presentation-quality client-side canvas viewer. The static backdrop is fetched
# once, snapshots are polled at 20 Hz and the metrics panel refreshes at 2 Hz
# while the canvas redraws at the display refresh rate.
#
# The presentation assets live in spatial_map_coop/viewer_assets/ and are inlined
# verbatim, so the JavaScript exercised by the offline node tests is byte-for-byte
# the JavaScript the browser receives. Display interpolation/prediction is
# presentation-only and is namespaced under display.* everywhere it appears.
@app.route("/api/spatial_map/viewer", methods=["GET"])
def get_live_viewer():
    try:
        html = _viewer_html()
    except (OSError, RuntimeError) as exc:
        return jsonify({"error": f"viewer assets unavailable: {exc}"}), 503
    return Response(html, mimetype="text/html")


def main() -> None:
    global CONFIG
    CONFIG = parse_args()
    _initialize_multi_ue_service()
    _initialize_presentation_contracts()
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

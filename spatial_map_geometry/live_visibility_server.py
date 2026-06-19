#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, Response, jsonify, request
from matplotlib.patches import Polygon

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_map_geometry.association import associate_objects
from spatial_map_geometry.demo_two_view_overlap import build_demo_maps
from spatial_map_geometry.geometry import (
    bounds,
    convex_polygon_intersection,
    object_footprint_polygon,
    overlap_area,
    sensor_fov_polygon,
)
from spatial_map_geometry.occlusion_reasoner import infer_overlap_disagreements
from spatial_map_geometry.schemas import LocalSensorMap, SensorPose2D, SpatialObject

try:
    import extract_traffic_lights as traffic_map
except Exception:  # pragma: no cover - CARLA may be unavailable on analysis hosts.
    traffic_map = None


app = Flask(__name__)

state_lock = threading.Lock()
render_lock = threading.Lock()
static_lock = threading.Lock()

latest_maps: Dict[str, LocalSensorMap] = {}
latest_received_at: Dict[str, float] = {}
pinned_streams: set[str] = set()
latest_png: Optional[bytes] = None
latest_png_at = 0.0
latest_render_error: Optional[str] = None

static_cache: Dict[str, object] = {
    "loaded_at": 0.0,
    "map_name": "CARLA",
    "roads": [],
    "buildings": [],
    "traffic_lights": [],
    "error": None,
}

CONFIG: Optional[argparse.Namespace] = None

OBJECT_COLORS = {
    "vehicle": "#00d1ff",
    "person": "#ff5fd1",
    "pedestrian": "#ff5fd1",
    "cyclist": "#8aff80",
    "unknown": "#ffffff",
}

STREAM_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
]


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


def _stream_color(stream_id: str) -> str:
    index = abs(hash(stream_id)) % len(STREAM_COLORS)
    return STREAM_COLORS[index]


def _object_color(class_name: str) -> str:
    return OBJECT_COLORS.get(str(class_name).lower(), OBJECT_COLORS["unknown"])


def _normalize_pose(payload: Dict[str, object]) -> SensorPose2D:
    pose = payload.get("pose")
    if not isinstance(pose, dict):
        pose = payload.get("sensor_pose")
    if not isinstance(pose, dict):
        pose = {}
    return SensorPose2D(
        x=_safe_float(pose.get("x", payload.get("sensor_x"))),
        y=_safe_float(pose.get("y", payload.get("sensor_y"))),
        z=_safe_float(pose.get("z", payload.get("sensor_z"))),
        yaw_deg=_safe_float(
            pose.get("yaw_deg", pose.get("yaw", payload.get("sensor_yaw_deg", payload.get("sensor_yaw"))))
        ),
    )


def _normalize_fov_polygon(
    payload: Dict[str, object],
    pose: SensorPose2D,
    fov_deg: float,
    range_m: float,
) -> List[Tuple[float, float]]:
    raw = payload.get("fov_polygon")
    if raw is None:
        raw = payload.get("visibility_polygon")
    points: List[Tuple[float, float]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                points.append((_safe_float(item.get("x")), _safe_float(item.get("y"))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((_safe_float(item[0]), _safe_float(item[1])))
    if len(points) >= 3:
        return points
    return sensor_fov_polygon(pose, fov_deg=fov_deg, range_m=range_m)


def _normalize_object(raw: Dict[str, object], stream_id: str, frame_id: Optional[int], index: int) -> SpatialObject:
    location = raw.get("location") if isinstance(raw.get("location"), dict) else {}
    dimensions = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    class_name = str(raw.get("class_name") or raw.get("type") or raw.get("label") or "unknown").lower()
    if class_name in {"vehicle", "movingvehicle", "parkedvehicle"}:
        class_name = "vehicle"
    elif class_name in {"pedestrian", "walker"}:
        class_name = "person"
    return SpatialObject(
        object_id=str(raw.get("object_id") or raw.get("id") or f"{stream_id}:{frame_id}:{index}"),
        class_name=class_name,
        x=_safe_float(location.get("x", raw.get("x", raw.get("world_x")))),
        y=_safe_float(location.get("y", raw.get("y", raw.get("world_y")))),
        z=_safe_float(location.get("z", raw.get("z", raw.get("world_z")))),
        length=max(0.05, _safe_float(dimensions.get("length", raw.get("length", raw.get("size_x"))), 1.0)),
        width=max(0.05, _safe_float(dimensions.get("width", raw.get("width", raw.get("size_y"))), 1.0)),
        height=max(0.05, _safe_float(dimensions.get("height", raw.get("height", raw.get("size_z"))), 1.0)),
        yaw_deg=_safe_float(raw.get("yaw_deg", raw.get("yaw")), 0.0),
        confidence=max(0.0, min(1.0, _safe_float(raw.get("confidence", raw.get("score")), 1.0))),
        source_stream_id=stream_id,
        frame_id=frame_id,
        timestamp_s=None,
        metadata={k: v for k, v in raw.items() if k not in {"location", "dimensions"}},
    )


def _normalize_local_map(payload: Dict[str, object]) -> LocalSensorMap:
    stream_id = str(payload.get("stream_id") or payload.get("node_id") or "sensor_stream")
    frame_id = payload.get("frame_id")
    frame_id_int = None if frame_id is None else _safe_int(frame_id, 0)
    timestamp_s = payload.get("timestamp_s", payload.get("timestamp"))
    timestamp_value = None if timestamp_s is None else _safe_float(timestamp_s, time.time())
    fov_deg = _safe_float(payload.get("fov_deg", payload.get("camera_fov_deg")), 90.0)
    range_m = _safe_float(payload.get("range_m", payload.get("sensor_range_m")), 60.0)
    pose = _normalize_pose(payload)
    fov_polygon = _normalize_fov_polygon(payload, pose, fov_deg=fov_deg, range_m=range_m)
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raw_objects = payload.get("detections")
    if not isinstance(raw_objects, list):
        raw_objects = []
    objects = [
        _normalize_object(obj, stream_id=stream_id, frame_id=frame_id_int, index=index)
        for index, obj in enumerate(raw_objects)
        if isinstance(obj, dict)
    ]
    return LocalSensorMap(
        stream_id=stream_id,
        pose=pose,
        fov_polygon=fov_polygon,
        objects=objects,
        timestamp_s=timestamp_value,
        frame_id=frame_id_int,
        sensor_type=str(payload.get("sensor_type") or "rgb_radar"),
        fov_deg=fov_deg,
        range_m=range_m,
        provenance=payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {},
    )


def _active_maps(now: Optional[float] = None) -> List[LocalSensorMap]:
    cfg = _config()
    now = time.time() if now is None else float(now)
    with state_lock:
        maps = []
        for stream_id, local_map in latest_maps.items():
            if stream_id in pinned_streams:
                maps.append(local_map)
                continue
            age = now - float(latest_received_at.get(stream_id, 0.0))
            if age <= float(cfg.stream_stale_s):
                maps.append(local_map)
        return maps


def _load_static_map_context() -> Dict[str, object]:
    cfg = _config()
    if bool(cfg.no_carla_static_map) or traffic_map is None:
        return dict(static_cache)
    now = time.time()
    with static_lock:
        cache_age_s = now - float(static_cache.get("loaded_at", 0.0))
        cache_ready = float(static_cache.get("loaded_at", 0.0)) > 0.0
        if cache_ready and cache_age_s < float(cfg.static_map_refresh_s):
            return dict(static_cache)
        try:
            client = traffic_map.carla.Client(str(cfg.carla_host), int(cfg.carla_port))
            client.set_timeout(float(cfg.carla_timeout_s))
            world = client.get_world()
            carla_map = world.get_map()
            roads, buildings = traffic_map._build_precise_static_map(world, carla_map)
            traffic_lights = []
            for actor in sorted(world.get_actors().filter("traffic.traffic_light"), key=lambda item: item.id):
                loc = actor.get_location()
                traffic_lights.append(
                    {
                        "id": int(actor.id),
                        "location": {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)},
                    }
                )
            static_cache.update(
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
            static_cache["loaded_at"] = now
            static_cache["error"] = str(exc)
        return dict(static_cache)


def _draw_static_map(ax: "plt.Axes", static_context: Dict[str, object]) -> None:
    for building in static_context.get("buildings") or []:
        footprint = building.get("footprint") if isinstance(building, dict) else None
        if not footprint:
            continue
        ax.add_patch(
            Polygon(
                [(point["x"], point["y"]) for point in footprint],
                closed=True,
                facecolor="#252a31",
                edgecolor="#3a424d",
                alpha=0.9,
                zorder=1,
            )
        )
    for polyline in static_context.get("roads") or []:
        if len(polyline) < 2:
            continue
        if traffic_map is not None:
            try:
                polyline = traffic_map._smooth_polyline_for_plot(
                    polyline,
                    traffic_map.ROAD_CENTERLINE_SMOOTHING_PASSES,
                )
            except Exception:
                pass
        ax.plot(
            [point[0] for point in polyline],
            [point[1] for point in polyline],
            color="#59616d",
            linewidth=1.4,
            alpha=0.75,
            zorder=2,
        )
    lights = static_context.get("traffic_lights") or []
    if lights:
        xs = [_safe_float(item.get("location", {}).get("x")) for item in lights if isinstance(item, dict)]
        ys = [_safe_float(item.get("location", {}).get("y")) for item in lights if isinstance(item, dict)]
        ax.scatter(xs, ys, c="#ffcc66", s=32, marker="^", edgecolors="#111820", linewidths=0.4, zorder=3)


def _draw_visibility(ax: "plt.Axes", maps: Sequence[LocalSensorMap]) -> None:
    for local_map in maps:
        color = _stream_color(local_map.stream_id)
        ax.add_patch(
            Polygon(
                local_map.fov_polygon,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.14,
                linewidth=1.5,
                zorder=4,
                label=f"{local_map.stream_id} visible FoV",
            )
        )
        ax.scatter([local_map.pose.x], [local_map.pose.y], c=color, marker="^", s=100, edgecolors="white", zorder=7)
        ax.annotate(local_map.stream_id, (local_map.pose.x, local_map.pose.y), xytext=(5, 5), textcoords="offset points", color=color, fontsize=8)

    for i, first in enumerate(maps):
        for second in maps[i + 1 :]:
            overlap = convex_polygon_intersection(first.fov_polygon, second.fov_polygon)
            if len(overlap) >= 3:
                ax.add_patch(
                    Polygon(
                        overlap,
                        closed=True,
                        facecolor="#5eead4",
                        edgecolor="#99f6e4",
                        alpha=0.18,
                        linewidth=0.9,
                        zorder=5,
                        label="FoV overlap",
                    )
                )


def _draw_objects(ax: "plt.Axes", maps: Sequence[LocalSensorMap]) -> None:
    for local_map in maps:
        stream_color = _stream_color(local_map.stream_id)
        for obj in local_map.objects:
            color = _object_color(obj.class_name)
            ax.add_patch(
                Polygon(
                    object_footprint_polygon(obj),
                    closed=True,
                    facecolor=color,
                    edgecolor=stream_color,
                    alpha=0.58,
                    linewidth=1.0,
                    linestyle="-",
                    zorder=8,
                )
            )
            ax.scatter([obj.x], [obj.y], c=color, s=28, edgecolors="white", linewidths=0.5, zorder=9)


def _draw_associations(ax: "plt.Axes", associations: Sequence[object]) -> None:
    for association in associations:
        members = getattr(association, "members", [])
        if len(members) < 2:
            continue
        cx = float(getattr(association, "centroid_x", 0.0))
        cy = float(getattr(association, "centroid_y", 0.0))
        ax.scatter([cx], [cy], c="#ffffff", s=36, marker="x", linewidths=1.1, zorder=10)
        for member in members:
            ax.plot([member.x, cx], [member.y, cy], color="#ffffff", alpha=0.32, linewidth=0.8, zorder=9)


def _draw_hypotheses(ax: "plt.Axes", hypotheses: Sequence[object]) -> None:
    for hypothesis in hypotheses:
        x = float(getattr(hypothesis, "x", 0.0))
        y = float(getattr(hypothesis, "y", 0.0))
        ax.scatter([x], [y], c="#ff4d4d", s=90, marker="*", edgecolors="white", linewidths=0.7, zorder=12)
        ax.annotate(
            "possible occlusion",
            (x, y),
            xytext=(7, -10),
            textcoords="offset points",
            color="#ffb4b4",
            fontsize=8,
            zorder=12,
        )


def _view_bounds(maps: Sequence[LocalSensorMap], static_context: Dict[str, object]) -> Tuple[float, float, float, float]:
    points: List[Tuple[float, float]] = []
    for local_map in maps:
        points.extend(local_map.fov_polygon)
        points.append((local_map.pose.x, local_map.pose.y))
        points.extend(obj.xy for obj in local_map.objects)
    if points:
        min_x, min_y, max_x, max_y = bounds(points)
        pad = max(8.0, float(_config().view_padding_m))
        return (min_x - pad, min_y - pad, max_x + pad, max_y + pad)

    road_points: List[Tuple[float, float]] = []
    for polyline in static_context.get("roads") or []:
        road_points.extend((float(point[0]), float(point[1])) for point in polyline)
    if road_points:
        return bounds(road_points)
    return (-50.0, -50.0, 50.0, 50.0)


def _snapshot() -> Dict[str, object]:
    now = time.time()
    maps = _active_maps(now)
    associations = associate_objects(maps, distance_threshold_m=float(_config().association_distance_m))
    hypotheses = infer_overlap_disagreements(
        maps,
        distance_threshold_m=float(_config().association_distance_m),
        min_overlap_area_m2=float(_config().min_overlap_area_m2),
    )
    with state_lock:
        stream_ages = {
            stream_id: now - float(received_at)
            for stream_id, received_at in latest_received_at.items()
        }
    return {
        "schema": "scenesense.visibility_spatial_map.v0",
        "status": "active" if maps else "waiting_for_local_maps",
        "generated_at": now,
        "active_stream_count": len(maps),
        "active_maps": [item.to_dict() for item in maps],
        "stream_ages_s": stream_ages,
        "associations": [item.to_dict() for item in associations],
        "occlusion_hypotheses": [item.to_dict() for item in hypotheses],
        "config": {
            "stream_stale_s": float(_config().stream_stale_s),
            "render_interval_ms": int(_config().render_interval_ms),
            "association_distance_m": float(_config().association_distance_m),
            "min_overlap_area_m2": float(_config().min_overlap_area_m2),
        },
    }


def _render_png() -> bytes:
    snapshot = _snapshot()
    maps = _active_maps()
    associations = associate_objects(maps, distance_threshold_m=float(_config().association_distance_m))
    hypotheses = infer_overlap_disagreements(
        maps,
        distance_threshold_m=float(_config().association_distance_m),
        min_overlap_area_m2=float(_config().min_overlap_area_m2),
    )
    static_context = _load_static_map_context()
    min_x, min_y, max_x, max_y = _view_bounds(maps, static_context)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(13, 10))
        _draw_static_map(ax, static_context)
        _draw_visibility(ax, maps)
        _draw_objects(ax, maps)
        _draw_associations(ax, associations)
        _draw_hypotheses(ax, hypotheses)

        if not maps:
            ax.text(
                0.5,
                0.5,
                "Waiting for local sensor maps",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#d8dee9",
                fontsize=16,
            )
        if static_context.get("error"):
            ax.text(
                0.5,
                0.98,
                f"Static-map fallback: {static_context['error']}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                color="#ffd166",
                fontsize=8,
            )

        ax.set_title(
            f"SceneSense Visibility-Aware Spatial Map - {snapshot['status']}",
            fontsize=16,
        )
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)
        ax.grid(True, linestyle="--", alpha=0.18)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
        fig.tight_layout()
        out = io.BytesIO()
        fig.savefig(out, format="png", dpi=140)
        plt.close(fig)
        out.seek(0)
        return out.getvalue()


def _cached_render_png(force: bool = False) -> bytes:
    global latest_png, latest_png_at, latest_render_error
    interval_s = max(0.01, float(_config().render_interval_ms) / 1000.0)
    now = time.time()
    with render_lock:
        if not force and latest_png is not None and now - latest_png_at < interval_s:
            return latest_png
        try:
            latest_png = _render_png()
            latest_png_at = now
            latest_render_error = None
            return latest_png
        except Exception as exc:
            latest_render_error = str(exc)
            if latest_png is not None:
                return latest_png
            raise


@app.route("/healthz", methods=["GET"])
def healthz():
    with state_lock:
        streams = list(latest_maps.keys())
        pinned = sorted(pinned_streams)
    return jsonify({"status": "ok", "streams": streams, "pinned_streams": pinned, "render_error": latest_render_error})


@app.route("/api/local_maps/update", methods=["POST"])
def update_local_maps():
    payload = request.get_json(force=True, silent=False)
    raw_maps: Iterable[object]
    if isinstance(payload, dict) and isinstance(payload.get("local_maps"), list):
        raw_maps = payload["local_maps"]
    elif isinstance(payload, list):
        raw_maps = payload
    else:
        raw_maps = [payload]

    now = time.time()
    updated = []
    with state_lock:
        for raw in raw_maps:
            if not isinstance(raw, dict):
                continue
            local_map = _normalize_local_map(raw)
            latest_maps[local_map.stream_id] = local_map
            latest_received_at[local_map.stream_id] = now
            pinned_streams.discard(local_map.stream_id)
            updated.append(local_map.stream_id)
    return jsonify({"status": "ok", "updated_streams": updated})


@app.route("/api/local_maps/demo", methods=["POST", "GET"])
def load_demo_maps():
    now = time.time()
    demo_maps = build_demo_maps()
    with state_lock:
        for local_map in demo_maps:
            latest_maps[local_map.stream_id] = local_map
            latest_received_at[local_map.stream_id] = now
            if bool(_config().pin_demo_maps):
                pinned_streams.add(local_map.stream_id)
    return jsonify({"status": "ok", "loaded_streams": [item.stream_id for item in demo_maps]})


@app.route("/api/spatial_map/latest", methods=["GET"])
def latest_spatial_map():
    return jsonify(_snapshot())


@app.route("/api/spatial_map/live.png", methods=["GET"])
def live_png():
    force = request.args.get("force", "0") in {"1", "true", "yes"}
    png = _cached_render_png(force=force)
    response = Response(png, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/api/spatial_map/viewer", methods=["GET"])
def viewer():
    refresh_ms = max(500, _safe_int(request.args.get("refresh_ms"), int(_config().render_interval_ms)))
    status_refresh_ms = max(1000, min(5000, refresh_ms * 2))
    auto_demo = request.args.get("demo", "0") in {"1", "true", "yes"}
    return Response(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SceneSense Visibility Spatial Map</title>
  <style>
    body {{ margin: 0; background: #0b0f16; color: #e5e7eb; font-family: system-ui, sans-serif; }}
    header {{ min-height: 40px; display: flex; align-items: center; gap: 16px; padding: 8px 14px; background: #111827; flex-wrap: wrap; }}
    img {{ width: 100vw; height: calc(100vh - 56px); object-fit: contain; display: block; }}
    code {{ color: #93c5fd; }}
    button {{ background: #2563eb; color: white; border: 0; border-radius: 4px; padding: 6px 10px; cursor: pointer; }}
    #status {{ color: #cbd5e1; }}
    #status.warning {{ color: #fbbf24; }}
    #status.error {{ color: #f87171; }}
  </style>
</head>
<body>
  <header>
    <strong>SceneSense Visibility Spatial Map</strong>
    <span>refresh <code>{refresh_ms} ms</code></span>
    <span><code>/api/spatial_map/latest</code></span>
    <button id="load-demo" type="button">Load Demo Maps</button>
    <span id="status">checking status...</span>
  </header>
  <img id="map" alt="live spatial map">
  <script>
    const img = document.getElementById("map");
    const statusEl = document.getElementById("status");
    const loadDemoButton = document.getElementById("load-demo");
    const autoDemo = {"true" if auto_demo else "false"};
    let imageTimer = null;
    let imageInFlight = false;
    let statusInFlight = false;

    async function refreshStatus() {{
      if (statusInFlight) {{
        return;
      }}
      statusInFlight = true;
      try {{
        const [healthResponse, latestResponse] = await Promise.all([
          fetch("/healthz?t=" + Date.now()),
          fetch("/api/spatial_map/latest?t=" + Date.now())
        ]);
        const health = await healthResponse.json();
        const latest = await latestResponse.json();
        const streams = latest.active_stream_count || 0;
        const pinned = (health.pinned_streams || []).join(",") || "none";
        const renderError = health.render_error ? " render_error=" + health.render_error : "";
        statusEl.textContent = `status=${{latest.status}} streams=${{streams}} pinned=${{pinned}}${{renderError}}`;
        statusEl.className = streams > 0 ? "" : "warning";
        if (health.render_error) {{
          statusEl.className = "error";
        }}
      }} catch (error) {{
        statusEl.textContent = "status check failed: " + error;
        statusEl.className = "error";
      }} finally {{
        statusInFlight = false;
      }}
    }}

    function scheduleImageRefresh(delayMs = {refresh_ms}) {{
      if (imageTimer !== null) {{
        clearTimeout(imageTimer);
      }}
      imageTimer = setTimeout(refreshImage, delayMs);
    }}

    function refreshImage() {{
      if (imageInFlight) {{
        scheduleImageRefresh({refresh_ms});
        return;
      }}
      imageInFlight = true;
      img.src = "/api/spatial_map/live.png?t=" + Date.now();
    }}

    async function loadDemoMaps() {{
      statusEl.textContent = "loading demo maps...";
      await fetch("/api/local_maps/demo", {{ method: "POST" }});
      refreshStatus();
      refreshImage();
    }}

    img.addEventListener("load", () => {{
      imageInFlight = false;
      scheduleImageRefresh({refresh_ms});
    }});
    img.addEventListener("error", () => {{
      imageInFlight = false;
      scheduleImageRefresh(Math.max(1000, {refresh_ms}));
    }});
    loadDemoButton.addEventListener("click", loadDemoMaps);
    if (autoDemo) {{
      loadDemoMaps();
    }} else {{
      refreshStatus();
      refreshImage();
    }}
    setInterval(refreshStatus, {status_refresh_ms});
  </script>
</body>
</html>""",
        mimetype="text/html",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visibility-aware SceneSense spatial map prototype.")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=35021)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=2.0)
    parser.add_argument("--no-carla-static-map", action="store_true")
    parser.add_argument("--static-map-refresh-s", type=float, default=10.0)
    parser.add_argument("--stream-stale-s", type=float, default=2.5)
    parser.add_argument("--render-interval-ms", type=int, default=100)
    parser.add_argument("--association-distance-m", type=float, default=3.0)
    parser.add_argument("--min-overlap-area-m2", type=float, default=10.0)
    parser.add_argument("--view-padding-m", type=float, default=15.0)
    parser.add_argument("--load-demo-on-start", action="store_true")
    parser.add_argument(
        "--threaded-server",
        action="store_true",
        help="Use Flask's threaded development server. The default avoids request pile-up during expensive renders.",
    )
    parser.add_argument(
        "--no-pin-demo-maps",
        dest="pin_demo_maps",
        action="store_false",
        help="Let demo maps expire according to --stream-stale-s like live streams.",
    )
    parser.set_defaults(pin_demo_maps=True)
    return parser.parse_args()


def main() -> int:
    global CONFIG
    CONFIG = parse_args()
    if bool(CONFIG.load_demo_on_start):
        now = time.time()
        with state_lock:
            for local_map in build_demo_maps():
                latest_maps[local_map.stream_id] = local_map
                latest_received_at[local_map.stream_id] = now
                if bool(CONFIG.pin_demo_maps):
                    pinned_streams.add(local_map.stream_id)
    print(f"Viewer: http://127.0.0.1:{CONFIG.api_port}/api/spatial_map/viewer")
    print(f"Latest JSON: http://127.0.0.1:{CONFIG.api_port}/api/spatial_map/latest")
    app.run(
        host=str(CONFIG.api_host),
        port=int(CONFIG.api_port),
        threaded=bool(CONFIG.threaded_server),
        use_reloader=False,
        debug=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

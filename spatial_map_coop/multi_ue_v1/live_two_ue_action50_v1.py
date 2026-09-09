#!/usr/bin/env python3
"""Headless live two-UE/action-50 cooperative-map demonstration.

The primary mode owns two moving CARLA vehicles and their RGB/radar sensors;
the optional passive mode attaches sensors to two caller-owned vehicles. Both
logical UEs share one resident localhost SplitFusion runtime. This demonstrates
model-to-map integration, not two-device latency.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository has a namespace-level Phase-10 package and an older nested
# fusion package. Both are required by the frozen runtime and radar builder;
# extend the namespace exactly as the qualified Route-B adapter does.
import pole_lraspp_multimodal_fusion as _fusion_namespace

_LEGACY_FUSION_PACKAGE = (
    ROOT / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion"
).resolve()
if _LEGACY_FUSION_PACKAGE not in {
    Path(value).resolve() for value in _fusion_namespace.__path__
}:
    _fusion_namespace.__path__.append(str(_LEGACY_FUSION_PACKAGE))

from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from data_collection.radar_sweep_aggregator_v1 import (
    PREPARE_EVERY_N_TICKS,
    WORLD_DELTA_S,
    WORLD_TICK_HZ,
    RadarSweepAggregator,
    RadarSweepError,
)
from pole_lraspp_multimodal_fusion.radar_fusion import build_radar_sample

from .association import AssociationPolicy
from .contracts import MultiUEContractError
from .faults import FAULT_CASES, deterministic_fault_case, mutate_edge_result
from .service import MultiUESpatialMapService


SCHEMA = "scenesense.live_two_ue_action50_demo.v1"
EDGE_RESULT_SCHEMA = "splitfusion_edge_result.v2"
OBJECT_MAP_UPDATE_SCHEMA = "splitfusion_object_map_update.v1"
EXECUTE_TOKEN = "SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO"
COMPLETE_TERMINAL = "SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO_COMPLETE"
ACTION_ID = 50
EXPECTED_PROFILE = ("split_ae64_uint4_q5000", "AE64", "UINT4", 5000, 10752)
CLOCK_DOMAIN = "carla_simulation_ns"
CAMERA_MOUNT = (1.8, 0.0, 1.55, -4.0, 0.0, 0.0)
RADAR_MOUNT = (2.0, 0.0, 1.0, 0.0, 0.0, 0.0)
MODEL_SIZE = (768, 448)


class LiveTwoUEError(RuntimeError):
    """The live demonstration contract or execution failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveTwoUEError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _append_json_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    handle.write("\n")
    handle.flush()


def _policy() -> AssociationPolicy:
    return AssociationPolicy(
        maximum_observation_age_ns=500_000_000,
        alignment_tolerance_ns=100_000_000,
        maximum_pair_time_delta_ns=100_000_000,
        maximum_xy_distance_m=4.0,
        maximum_relative_size_difference=0.65,
        track_match_distance_m=6.0,
        track_stale_after_ns=4_000_000_000,
        minimum_confidence=0.0,
    )


def _runtime_api() -> Any:
    """Load the frozen live stack only for preflight/execution, never listing."""

    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime

    _require(
        live_pilot_runtime.EDGE_RESULT_SCHEMA == EDGE_RESULT_SCHEMA
        and live_pilot_runtime.OBJECT_MAP_UPDATE_SCHEMA == OBJECT_MAP_UPDATE_SCHEMA,
        "SplitFusion result schema drift",
    )
    return live_pilot_runtime


def _profile_document(profile: Any) -> dict[str, Any]:
    return {
        "action_id": int(profile.action_id),
        "profile_id": str(profile.profile_id),
        "family": str(profile.family),
        "quantizer": str(profile.quantizer),
        "q": float(profile.q),
        "q_e4": int(profile.q_e4),
        "keep_count": int(profile.keep_count),
        "routing_tag": int(profile.routing_tag),
        "zstd_level": int(profile.zstd_level),
        "wire_layout": str(profile.wire.layout),
        "segmentation_installable": bool(profile.segmentation_installable),
    }


def _verify_profile(registry: SplitActionRegistry) -> Any:
    profile = registry.resolve(ACTION_ID)
    observed = (
        profile.profile_id,
        profile.family,
        profile.quantizer,
        profile.q_e4,
        profile.keep_count,
    )
    _require(observed == EXPECTED_PROFILE, f"action-50 identity drift: {observed!r}")
    _require(profile.zstd_level == 1, "action 50 must retain zstd level 1")
    _require(profile.wire.layout == "CURRENT_CELL_MAJOR", "action 50 layout drift")
    return profile


def _sensor_transform(values: tuple[float, float, float, float, float, float]) -> Any:
    import carla

    x, y, z, pitch, yaw, roll = values
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )


def _camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    return np.asarray(
        ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _offset_transform(transform: Any, *, forward_m: float, z_m: float = 0.0) -> Any:
    """Return a CARLA transform translated in the actor's forward direction."""

    import carla

    forward = transform.get_forward_vector()
    location = transform.location
    return carla.Transform(
        carla.Location(
            x=float(location.x) + float(forward.x) * float(forward_m),
            y=float(location.y) + float(forward.y) * float(forward_m),
            z=float(location.z) + float(z_m),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def _fixed_route(world: Any, indices_text: str, spacing_m: float = 2.0) -> list[Any]:
    """Build the existing Stage-2 route without importing its inference client."""

    indices = [int(value.strip()) for value in str(indices_text).split(",") if value.strip()]
    if not indices:
        return []
    spawn_points = list(world.get_map().get_spawn_points())
    invalid = [value for value in indices if value < 0 or value >= len(spawn_points)]
    _require(not invalid, f"invalid fixed-route spawn indices: {invalid}")
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError:
        GlobalRoutePlanner = None

    key_points = [spawn_points[index].location for index in indices]
    if GlobalRoutePlanner is None or len(key_points) < 2:
        return key_points
    planner = GlobalRoutePlanner(world.get_map(), float(spacing_m))
    route: list[Any] = []
    for start, end in zip(key_points[:-1], key_points[1:]):
        trace = planner.trace_route(start, end)
        values = [waypoint.transform.location for waypoint, _option in trace]
        if not values:
            values = [start, end]
        for location in values:
            if not route or float(route[-1].distance(location)) >= float(spacing_m):
                route.append(location)
    return route


class _OwnedTwoEgoScenario:
    """A lightweight CARLA clock/actor owner with no duplicate model pipeline."""

    def __init__(self, client: Any, world: Any, args: argparse.Namespace) -> None:
        self.client = client
        self.world = world
        self.args = args
        self.traffic_manager = client.get_trafficmanager(int(args.tm_port))
        self.original_settings = world.get_settings()
        self.actors: list[Any] = []
        self.ego_a: Any | None = None
        self.ego_b: Any | None = None
        self.settings_changed = False
        self.tm_sync_enabled = False

    def _spawn_ego(self, transform: Any, role_name: str) -> Any:
        library = self.world.get_blueprint_library()
        matches = list(library.filter(str(self.args.ego_blueprint)))
        if not matches:
            matches = list(library.filter("vehicle.*"))
        _require(bool(matches), "CARLA exposes no vehicle blueprint")
        blueprint = library.find(matches[0].id)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", str(role_name))
        actor = self.world.try_spawn_actor(blueprint, transform)
        _require(actor is not None, f"failed to spawn {role_name}")
        self.actors.append(actor)
        return actor

    def start(self) -> tuple[Any, Any]:
        existing = list(self.world.get_actors().filter("vehicle.*"))
        existing += list(self.world.get_actors().filter("walker.pedestrian.*"))
        _require(
            not existing,
            "owned scenario requires a fresh CARLA world with no vehicles or pedestrians",
        )
        try:
            random.seed(int(self.args.scenario_seed))
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            # The registered radar tensor is built from two 100 ms logical
            # sweeps, each containing two callbacks from a 20 Hz CARLA sensor.
            # The model remains 10 Hz because only every second tick is used.
            settings.fixed_delta_seconds = WORLD_DELTA_S
            self.world.apply_settings(settings)
            self.settings_changed = True
            self.traffic_manager.set_synchronous_mode(True)
            self.tm_sync_enabled = True
            self.traffic_manager.set_random_device_seed(int(self.args.scenario_seed))
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            try:
                self.world.set_pedestrians_seed(int(self.args.scenario_seed))
            except (AttributeError, RuntimeError):
                pass

            spawn_points = list(self.world.get_map().get_spawn_points())
            _require(bool(spawn_points), "CARLA map has no vehicle spawn points")
            index = int(self.args.ego_spawn_index)
            _require(0 <= index < len(spawn_points), "ego spawn index is outside this map")
            base = spawn_points[index]
            self.ego_a = self._spawn_ego(
                _offset_transform(base, forward_m=0.0, z_m=0.15),
                "scenesense_action50_ue_a",
            )
            self.ego_b = self._spawn_ego(
                _offset_transform(base, forward_m=-float(self.args.ego_gap_m), z_m=0.15),
                "scenesense_action50_ue_b",
            )
            route = _fixed_route(self.world, self.args.fixed_route_spawn_indices)
            for ego in (self.ego_a, self.ego_b):
                ego.set_autopilot(True, int(self.args.tm_port))
                self.traffic_manager.ignore_lights_percentage(ego, 50.0)
                self.traffic_manager.vehicle_percentage_speed_difference(ego, 60.0)
                self.traffic_manager.distance_to_leading_vehicle(ego, 28.0)
                self.traffic_manager.auto_lane_change(ego, False)
                if route:
                    self.traffic_manager.set_path(ego, list(route))

            # Reuse the established CARLA-only population helpers. They do not
            # construct or execute either perception model.
            import carla_split_inference_udp_segmentation_trained_lraspp_pole_client as pole_client

            anchor = self.ego_a.get_location()
            vehicles = pole_client.spawn_background_vehicles_near(
                self.client,
                self.world,
                self.traffic_manager,
                anchor,
                int(self.args.npc_vehicles),
                float(self.args.spawn_radius_m),
            )
            walkers, controllers = pole_client.spawn_background_pedestrians_near(
                self.client,
                self.world,
                anchor,
                int(self.args.npc_pedestrians),
                float(self.args.spawn_radius_m),
            )
            self.actors.extend(vehicles)
            self.actors.extend(walkers)
            self.actors.extend(controllers)
            self.world.tick()
            print(
                "Owned two-ego scenario ready: "
                f"ue_a={self.ego_a.id}, ue_b={self.ego_b.id}, "
                f"vehicles={len(vehicles)}, pedestrians={len(walkers)}, "
                f"route_points={len(route)}, world_hz={WORLD_TICK_HZ:g}, "
                f"prepared_hz={WORLD_TICK_HZ / PREPARE_EVERY_N_TICKS:g}"
            )
            return self.ego_a, self.ego_b
        except BaseException:
            self.close()
            raise

    def tick(self) -> int:
        return int(self.world.tick())

    def close(self) -> None:
        actor_ids: list[int] = []
        for actor in reversed(self.actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except (AttributeError, RuntimeError):
                pass
            try:
                actor_ids.append(int(actor.id))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # Destroy the owned population in one server-side batch. Per-actor
        # destroy RPCs are both slow and prone to cascading native errors when
        # a long run is already unwinding. Reversed order puts walker
        # controllers before their walkers and both egos last.
        if actor_ids:
            try:
                import carla

                responses = self.client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                    True,
                )
                failures = [
                    f"{actor_id}: {response.error}"
                    for actor_id, response in zip(actor_ids, responses)
                    if getattr(response, "error", None)
                ]
                if failures:
                    print(
                        "Owned-scenario batch cleanup reported "
                        f"{len(failures)}/{len(actor_ids)} failures: "
                        + "; ".join(failures[:5])
                    )
            except (ImportError, RuntimeError) as exc:
                print(f"Owned-scenario batch cleanup could not be verified: {exc}")
        self.actors.clear()
        if self.tm_sync_enabled:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
            self.tm_sync_enabled = False
        if self.settings_changed:
            try:
                self.world.apply_settings(self.original_settings)
            except RuntimeError:
                pass
            self.settings_changed = False


def _image_bgr(image: Any) -> np.ndarray:
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)
    )[:, :, :3].copy()


class _PassiveUESensors:
    """Own only two sensors attached to a caller-owned CARLA vehicle."""

    def __init__(self, world: Any, vehicle: Any, stream_id: str) -> None:
        runtime = _runtime_api()
        self.world = world
        self.vehicle = vehicle
        self.stream_id = str(stream_id)
        self.condition = threading.Condition()
        self.images: "OrderedDict[int, tuple[Any, Any]]" = OrderedDict()
        self.radars: "OrderedDict[int, Any]" = OrderedDict()
        self.aggregator = RadarSweepAggregator(keep_sweeps=12)
        self.aggregator_error = ""
        self.image_callbacks = 0
        self.radar_callbacks = 0
        self.prepare_attempts = 0
        self.prepare_rejections: Counter[str] = Counter()
        self.tracker = runtime.FastStationaryTrackAccumulator(
            stationary_velocity_mps=0.35,
            parked_threshold_s=5.0,
            association_grid_m=1.5,
            max_stale_s=2.0,
        )
        self.intrinsics = _camera_intrinsics(MODEL_SIZE[0], MODEL_SIZE[1], 120.0)
        self.sensors: list[Any] = []
        self._spawn()

    @staticmethod
    def _prune(values: OrderedDict[int, Any]) -> None:
        while len(values) > 64:
            values.popitem(last=False)

    def _spawn(self) -> None:
        blueprints = self.world.get_blueprint_library()
        rgb = blueprints.find("sensor.camera.rgb")
        rgb.set_attribute("image_size_x", "1280")
        rgb.set_attribute("image_size_y", "720")
        rgb.set_attribute("fov", "120")
        rgb.set_attribute("sensor_tick", "0.0")
        radar = blueprints.find("sensor.other.radar")
        radar.set_attribute("range", "120")
        radar.set_attribute("horizontal_fov", "120")
        radar.set_attribute("vertical_fov", "30")
        radar.set_attribute("points_per_second", "200000")
        radar.set_attribute("sensor_tick", "0.0")
        try:
            camera_actor = self.world.spawn_actor(
                rgb, _sensor_transform(CAMERA_MOUNT), attach_to=self.vehicle
            )
            self.sensors.append(camera_actor)
            radar_actor = self.world.spawn_actor(
                radar, _sensor_transform(RADAR_MOUNT), attach_to=self.vehicle
            )
            self.sensors.append(radar_actor)
            camera_actor.listen(self._on_image)
            radar_actor.listen(self._on_radar)
        except BaseException:
            self.close()
            raise

    def _on_image(self, image: Any) -> None:
        try:
            ego_transform = self.vehicle.get_transform()
        except RuntimeError:
            return
        with self.condition:
            self.image_callbacks += 1
            self.images[int(image.frame)] = (image, ego_transform)
            self._prune(self.images)
            self.condition.notify_all()

    def _on_radar(self, radar: Any) -> None:
        with self.condition:
            self.radar_callbacks += 1
            try:
                self.aggregator.ingest(radar)
            except Exception as exc:
                self.aggregator_error = f"{type(exc).__name__}: {exc}"
            self.radars[int(radar.frame)] = radar
            self._prune(self.radars)
            self.condition.notify_all()

    def ready_frames(self, after_frame: int) -> set[int]:
        with self.condition:
            return {
                frame for frame in self.images.keys() & self.radars.keys()
                if frame > int(after_frame)
            }

    def prepare(self, frame_id: int) -> dict[str, Any] | None:
        with self.condition:
            self.prepare_attempts += 1
            _require(not self.aggregator_error, self.aggregator_error)
            image_item = self.images.get(int(frame_id))
            radar = self.radars.get(int(frame_id))
            if image_item is None or radar is None:
                self.prepare_rejections["MISSING_COMMON_SENSOR_FRAME"] += 1
                return None
            if self.aggregator.anchor_s is None:
                self.aggregator.set_anchor(float(radar.timestamp))
            sweep_index = self.aggregator.sweep_index_for(float(radar.timestamp))
            if not self.aggregator.has_window(sweep_index):
                self.prepare_rejections["INCOMPLETE_SWEEP_WINDOW"] += 1
                return None
            radar_inverse = np.asarray(radar.transform.get_inverse_matrix(), dtype=np.float64)
            try:
                detections, window = self.aggregator.window_detections(
                    sweep_index,
                    sensor_inverse_matrix=radar_inverse,
                    reference_timestamp_s=float(radar.timestamp),
                )
            except RadarSweepError:
                self.prepare_rejections["RADAR_SWEEP_ERROR"] += 1
                return None
        if int(window["callbacks"]) != 4:
            self.prepare_rejections[
                f"WINDOW_CALLBACK_COUNT_{int(window['callbacks'])}"
            ] += 1
            return None
        image, ego_transform = image_item
        radar_matrix = np.asarray(radar.transform.get_matrix(), dtype=np.float64)
        camera_inverse = np.asarray(image.transform.get_inverse_matrix(), dtype=np.float64)
        tensor, _points, summary = build_radar_sample(
            detections=detections,
            sensor_matrix=radar_matrix,
            camera_inverse_matrix=camera_inverse,
            camera_intrinsics=self.intrinsics,
            width=MODEL_SIZE[0],
            height=MODEL_SIZE[1],
            frame_time_s=float(radar.timestamp),
            tracker=self.tracker,
            max_range_m=120.0,
            max_abs_velocity_mps=20.0,
            parked_threshold_s=5.0,
            point_radius_px=4,
            rasterizer="fast",
        )
        return {
            "frame_bgr": _image_bgr(image),
            "radar_tensor": tensor,
            "radar_summary": summary,
            "window": window,
            "ego_transform": ego_transform,
            "capture_timestamp_ns": int(round(float(image.timestamp) * 1_000_000_000)),
            "carla_timestamp_s": float(image.timestamp),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return compact sensor evidence suitable for a failed-run record."""

        with self.condition:
            sweep_reports = self.aggregator.all_sweep_reports()
            return {
                "stream_id": self.stream_id,
                "image_callbacks": int(self.image_callbacks),
                "radar_callbacks": int(self.radar_callbacks),
                "common_buffered_frames": len(self.images.keys() & self.radars.keys()),
                "prepare_attempts": int(self.prepare_attempts),
                "prepare_rejections": dict(sorted(self.prepare_rejections.items())),
                "aggregator_error": self.aggregator_error or None,
                "radar_raw_callbacks": int(self.aggregator.raw_callbacks),
                "radar_dropped_callback_frames": int(
                    self.aggregator.dropped_callback_frames
                ),
                "expected_callbacks_per_sweep": int(
                    self.aggregator.expected_callbacks_per_sweep
                ),
                "expected_window_callbacks": int(
                    self.aggregator.expected_window_callbacks
                ),
                "recent_sweeps": sweep_reports[-6:],
            }

    def close(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.stop()
            except (RuntimeError, AttributeError):
                pass
        for sensor in reversed(self.sensors):
            try:
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()


def _edge_payload(result: Any, snapshot: Any) -> dict[str, Any]:
    context = result.metadata.frame_context
    _require(context is not None, "edge result lacks SFD1-v2 frame context")
    records = list(snapshot.records or ())
    return {
        "schema": EDGE_RESULT_SCHEMA,
        "action_id": int(result.metadata.action_id),
        "profile_id": str(result.metadata.profile_id),
        "stream_id": str(context.stream_id),
        "frame_id": int(context.frame_id),
        "capture_timestamp_ns": int(context.capture_timestamp_ns),
        "object_map_update": {
            "schema": OBJECT_MAP_UPDATE_SCHEMA,
            "stream_id": str(context.stream_id),
            "frame_id": int(context.frame_id),
            "capture_timestamp_ns": int(context.capture_timestamp_ns),
            "action_id": int(result.metadata.action_id),
            "records": records,
        },
    }


def _ego_pose_document(
    *, ue_id: str, source: _PassiveUESensors, prepared_sensor: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain the compact, measured pose needed for an ego-following map view."""

    transform = prepared_sensor["ego_transform"]
    location, rotation = transform.location, transform.rotation
    return {
        "ue_id": ue_id,
        "actor_id": int(source.vehicle.id),
        "stream_id": source.stream_id,
        "capture_timestamp_ns": int(prepared_sensor["capture_timestamp_ns"]),
        "location": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        },
        "rotation": {
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
            "roll": float(rotation.roll),
        },
    }


def _source_observation_document(
    *, ue_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain compact model reports, never CARLA actors or raw sensor data."""

    update = payload["object_map_update"]
    return {
        "ue_id": ue_id,
        "stream_id": str(payload["stream_id"]),
        "frame_id": int(payload["frame_id"]),
        "capture_timestamp_ns": int(payload["capture_timestamp_ns"]),
        "action_id": int(payload["action_id"]),
        "records": list(update["records"]),
    }


def _run_one(
    *,
    source: _PassiveUESensors,
    prepared_sensor: Mapping[str, Any],
    ue_runtime: Any,
    edge_runtime: Any,
    tail: Any,
    device: Any,
    frame_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    runtime = _runtime_api()
    transform = prepared_sensor["ego_transform"]
    location, rotation = transform.location, transform.rotation
    capture_ns = int(prepared_sensor["capture_timestamp_ns"])
    sequence_id = (int(source.vehicle.id) << 32) | int(frame_id)
    context = build_frame_context_v1(
        stream_id=source.stream_id,
        frame_id=int(frame_id),
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_ns,
        ego_world_x=float(location.x),
        ego_world_y=float(location.y),
        ego_world_z=float(location.z),
        ego_world_pitch=float(rotation.pitch),
        ego_world_yaw=float(rotation.yaw),
        ego_world_roll=float(rotation.roll),
    )
    started_ns = time.perf_counter_ns()
    input_7ch = runtime._prepare_live_input(
        prepared_sensor["frame_bgr"], prepared_sensor["radar_tensor"], device
    )
    with torch.inference_mode():
        encoded = ue_runtime.prepare(
            ACTION_ID,
            input_7ch,
            sequence_id=sequence_id,
            capture_timestamp_ns=capture_ns,
            frame_context=context,
        )
        edge = edge_runtime.process(encoded.wire_bytes, transmitted_action_id=ACTION_ID)
    snapshot = tail.take_snapshot()
    finished_ns = time.perf_counter_ns()
    payload = _edge_payload(edge, snapshot)
    return payload, {
        "ue_id": source.stream_id,
        "actor_id": int(source.vehicle.id),
        "frame_id": int(frame_id),
        "capture_timestamp_ns": capture_ns,
        "carla_timestamp_s": float(prepared_sensor["carla_timestamp_s"]),
        "object_count": len(payload["object_map_update"]["records"]),
        "scientific_inner_bytes": int(encoded.inner_payload_bytes),
        "sfd1_bytes": int(encoded.total_transmitted_bytes),
        "compact_object_json_bytes": len(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "localhost_model_path_ms": (finished_ns - started_ns) / 1_000_000.0,
        "ue_timing_ns": runtime._trace_ns(encoded.timing),
        "edge_timing_ns": runtime._trace_ns(edge.timing),
        "radar_callbacks": int(prepared_sensor["window"]["callbacks"]),
        "radar_returns": int(prepared_sensor["window"]["returns"]),
    }


def _shadow_case(
    *,
    case: str,
    payload_a: Mapping[str, Any],
    payload_b: Mapping[str, Any],
    timestamp_ns: int,
    session_id: str,
) -> dict[str, Any]:
    records_b = payload_b.get("object_map_update", {}).get("records", [])
    stale_timestamp_too_early = (
        case == "STALE"
        and int(payload_b.get("capture_timestamp_ns", 0)) < 1_000_000_000
    )
    if (case != "DUPLICATE_IDENTICAL" and not records_b) or stale_timestamp_too_early:
        return {
            "case": case,
            "applicable": False,
            "not_applicable_reason": (
                "CARLA_TIMESTAMP_TOO_EARLY_FOR_REGISTERED_STALE_OFFSET"
                if stale_timestamp_too_early
                else "UE_B_OBJECT_SET_EMPTY"
            ),
            "ingest_dispositions": [],
            "ingress_error": "",
            "accepted_observations": 0,
            "rejected_observations": 0,
            "rejection_reasons": [],
            "association_count": 0,
            "multi_source_associations": 0,
            "track_count": 0,
        }
    service = MultiUESpatialMapService(_policy())
    first = service.ingest_splitfusion(
        payload_a,
        ue_id="ue-a",
        session_id=session_id,
        received_timestamp_ns=timestamp_ns,
        clock_domain=CLOCK_DOMAIN,
    )
    dispositions = [first.disposition]
    error = ""
    try:
        if case == "IDENTITY_CONFLICT":
            nominal = service.ingest_splitfusion(
                payload_b,
                ue_id="ue-b",
                session_id=session_id,
                received_timestamp_ns=timestamp_ns,
                clock_domain=CLOCK_DOMAIN,
            )
            dispositions.append(nominal.disposition)
            bad = mutate_edge_result(payload_b, case)
            service.ingest_splitfusion(
                bad,
                ue_id="ue-b",
                session_id=session_id,
                received_timestamp_ns=timestamp_ns,
                clock_domain=CLOCK_DOMAIN,
            )
        else:
            bad = mutate_edge_result(payload_b, case)
            result = service.ingest_splitfusion(
                bad,
                ue_id="ue-b",
                session_id=session_id,
                received_timestamp_ns=timestamp_ns,
                clock_domain=CLOCK_DOMAIN,
            )
            dispositions.append(result.disposition)
            if case == "DUPLICATE_IDENTICAL":
                duplicate = service.ingest_splitfusion(
                    bad,
                    ue_id="ue-b",
                    session_id=session_id,
                    received_timestamp_ns=timestamp_ns,
                    clock_domain=CLOCK_DOMAIN,
                )
                dispositions.append(duplicate.disposition)
    except MultiUEContractError as exc:
        error = f"{type(exc).__name__}: {exc}"
    snapshot = service.snapshot(
        clock_domain=CLOCK_DOMAIN, snapshot_timestamp_ns=timestamp_ns
    ).as_dict()
    expected_rejection = case in ("IDENTITY_CONFLICT", "NONFINITE")
    _require(bool(error) == expected_rejection, f"{case} ingress behavior drift: {error!r}")
    if case == "DUPLICATE_IDENTICAL":
        _require(dispositions[-1] == "DUPLICATE_IDENTICAL", "duplicate was not idempotent")
    if case == "STALE":
        reasons = {item["reason"] for item in snapshot["rejected_observations"]}
        _require(bool(reasons & {"SOURCE_OUTSIDE_ALIGNMENT_WINDOW", "OBSERVATION_TOO_OLD"}),
                 "stale observation was not filtered")
    return {
        "case": case,
        "applicable": True,
        "not_applicable_reason": "",
        "ingest_dispositions": dispositions,
        "ingress_error": error,
        "accepted_observations": int(snapshot["accepted_observation_count"]),
        "rejected_observations": len(snapshot["rejected_observations"]),
        "rejection_reasons": sorted(
            {item["reason"] for item in snapshot["rejected_observations"]}
        ),
        "association_count": int(snapshot["association_count"]),
        "multi_source_associations": sum(
            int(item["source_count"] >= 2) for item in snapshot["associations"]
        ),
        "track_count": len(snapshot["tracks"]),
    }


def _vehicle_rows(world: Any) -> list[dict[str, Any]]:
    rows = []
    for actor in world.get_actors().filter("vehicle.*"):
        transform = actor.get_transform()
        rows.append(
            {
                "actor_id": int(actor.id),
                "type_id": str(actor.type_id),
                "role_name": str(actor.attributes.get("role_name", "")),
                "location": {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                },
            }
        )
    return sorted(rows, key=lambda row: row["actor_id"])


def _connect_carla(host: str, port: int, timeout_s: float) -> tuple[Any, Any]:
    import carla

    client = carla.Client(str(host), int(port))
    client.set_timeout(float(timeout_s))
    return client, client.get_world()


def _resolve_vehicle(world: Any, actor_id: int) -> Any:
    actor = world.get_actor(int(actor_id))
    _require(actor is not None, f"CARLA actor {actor_id} does not exist")
    _require(str(actor.type_id).startswith("vehicle."), f"actor {actor_id} is not a vehicle")
    return actor


def _require_actor_selection(args: argparse.Namespace) -> None:
    """Validate external IDs only when the harness does not own the egos."""

    if bool(args.spawn_two_egos):
        _require(
            int(args.ue_a_actor_id) < 0 and int(args.ue_b_actor_id) < 0,
            "owned two-ego mode cannot accept external actor IDs",
        )
        return
    _require(
        int(args.ue_a_actor_id) != int(args.ue_b_actor_id),
        "two distinct ego actor IDs are required",
    )


def _require_sensor_clock_contract(args: argparse.Namespace) -> None:
    """Prevent a valid CLI from silently violating the frozen radar cadence."""

    if bool(args.spawn_two_egos):
        _require(
            int(args.frame_stride) == PREPARE_EVERY_N_TICKS,
            "owned mode frame stride must equal the registered two-tick cadence",
        )


def _safe_sensor_diagnostics(source: _PassiveUESensors | None) -> dict[str, Any] | None:
    if source is None:
        return None
    try:
        return source.diagnostics()
    except (AttributeError, RuntimeError, ValueError) as exc:
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def _manifest(output: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    files = {}
    for name in ("observations.jsonl", "snapshots.jsonl", "faults.jsonl", "SUMMARY.json"):
        path = output / name
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return {
        "schema": f"{SCHEMA}.manifest",
        "summary_sha256": files["SUMMARY.json"]["sha256"],
        "files": files,
        "raw_rgb_frames_retained": 0,
        "raw_radar_frames_retained": 0,
        "semantic_masks_retained": 0,
        "ui_launched": False,
        "summary_counts": {
            "paired_frames": int(summary["paired_frames"]),
            "nominal_multi_source_associations": int(
                summary["nominal_multi_source_associations"]
            ),
        },
    }


def run(args: argparse.Namespace) -> int:
    import torch

    if args.list_vehicles:
        _client, world = _connect_carla(
            args.carla_host, args.carla_port, args.carla_timeout_s
        )
        print(json.dumps(_vehicle_rows(world), indent=2, sort_keys=True))
        return 0

    registry = SplitActionRegistry.from_runtime_binding()
    profile = _verify_profile(registry)
    runtime = _runtime_api()
    if args.preflight:
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(torch.cuda.device_count() >= 1, "no CUDA device is visible")
        print(json.dumps({
            "status": "PREFLIGHT_PASS",
            "profile": _profile_document(profile),
            "cuda_device": torch.cuda.get_device_name(0),
            "registry_audit": asdict(registry.startup_audit),
        }, indent=2, sort_keys=True))
        return 0

    client, world = _connect_carla(
        args.carla_host, args.carla_port, args.carla_timeout_s
    )

    _require(args.execute == EXECUTE_TOKEN, f"--execute must equal {EXECUTE_TOKEN}")
    _require_actor_selection(args)
    _require_sensor_clock_contract(args)
    _require(args.max_pairs >= len(FAULT_CASES), f"max-pairs must be >= {len(FAULT_CASES)}")
    output = Path(args.output).resolve()
    _require(not output.exists(), f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started_wall = time.time()
    source_a = source_b = None
    owned_scenario: _OwnedTwoEgoScenario | None = None
    pair_count = 0
    last_frame = -1
    try:
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        device = torch.device("cuda:0")
        if args.spawn_two_egos:
            owned_scenario = _OwnedTwoEgoScenario(client, world, args)
            vehicle_a, vehicle_b = owned_scenario.start()
        else:
            vehicle_a = _resolve_vehicle(world, args.ue_a_actor_id)
            vehicle_b = _resolve_vehicle(world, args.ue_b_actor_id)
        ue_runtime, ue_ledger, ue_models = runtime._preload_ue(device)
        edge_runtime, tail, edge_ledger, edge_models = runtime._preload_edge(device)
        del ue_models, edge_models
        # Load and freeze every model before sensors begin. Otherwise callbacks
        # accumulate an unbounded pre-anchor radar backlog during checkpoint IO.
        source_a = _PassiveUESensors(world, vehicle_a, "two-ue/ue-a")
        source_b = _PassiveUESensors(world, vehicle_b, "two-ue/ue-b")
        service = MultiUESpatialMapService(_policy())
        session_id = str(args.session_id)
        frame_parity: int | None = None
        nominal_multi_source = 0
        nominal_associations = 0
        compact_model_records = 0
        fault_counts: Counter[str] = Counter()
        fault_case_cursor = 0
        deadline = time.monotonic() + float(args.duration_s)
        with (
            (output / "observations.jsonl").open("x", encoding="utf-8") as observations,
            (output / "snapshots.jsonl").open("x", encoding="utf-8") as snapshots,
            (output / "faults.jsonl").open("x", encoding="utf-8") as faults,
        ):
            while pair_count < int(args.max_pairs) and time.monotonic() < deadline:
                if owned_scenario is not None:
                    tick_frame = owned_scenario.tick()
                    callback_deadline = time.monotonic() + float(args.sensor_timeout_s)
                    while time.monotonic() < callback_deadline:
                        if (
                            tick_frame in source_a.ready_frames(last_frame)
                            and tick_frame in source_b.ready_frames(last_frame)
                        ):
                            break
                        time.sleep(0.005)
                common = source_a.ready_frames(last_frame) & source_b.ready_frames(last_frame)
                if frame_parity is not None:
                    common = {frame for frame in common if frame % args.frame_stride == frame_parity}
                if not common:
                    time.sleep(0.01)
                    continue
                frame_id = max(common)
                if frame_parity is None:
                    frame_parity = frame_id % args.frame_stride
                prepared_a = source_a.prepare(frame_id)
                prepared_b = source_b.prepare(frame_id)
                last_frame = frame_id
                if prepared_a is None or prepared_b is None:
                    continue
                payload_a, row_a = _run_one(
                    source=source_a, prepared_sensor=prepared_a, ue_runtime=ue_runtime,
                    edge_runtime=edge_runtime, tail=tail, device=device, frame_id=frame_id,
                )
                payload_b, row_b = _run_one(
                    source=source_b, prepared_sensor=prepared_b, ue_runtime=ue_runtime,
                    edge_runtime=edge_runtime, tail=tail, device=device, frame_id=frame_id,
                )
                timestamp_ns = max(
                    int(payload_a["capture_timestamp_ns"]),
                    int(payload_b["capture_timestamp_ns"]),
                )
                result_a = service.ingest_splitfusion(
                    payload_a, ue_id="ue-a", session_id=session_id,
                    received_timestamp_ns=int(payload_a["capture_timestamp_ns"]),
                    clock_domain=CLOCK_DOMAIN,
                )
                result_b = service.ingest_splitfusion(
                    payload_b, ue_id="ue-b", session_id=session_id,
                    received_timestamp_ns=int(payload_b["capture_timestamp_ns"]),
                    clock_domain=CLOCK_DOMAIN,
                )
                snapshot = service.snapshot(
                    clock_domain=CLOCK_DOMAIN, snapshot_timestamp_ns=timestamp_ns
                ).as_dict()
                multi_source = sum(
                    int(item["source_count"] >= 2) for item in snapshot["associations"]
                )
                nominal_multi_source += multi_source
                nominal_associations += int(snapshot["association_count"])
                source_observations = [
                    _source_observation_document(ue_id="ue-a", payload=payload_a),
                    _source_observation_document(ue_id="ue-b", payload=payload_b),
                ]
                compact_model_records += sum(
                    len(source["records"]) for source in source_observations
                )
                _append_json_line(observations, {**row_a, "ingest": result_a.disposition})
                _append_json_line(observations, {**row_b, "ingest": result_b.disposition})
                _append_json_line(snapshots, {
                    "pair_index": pair_count,
                    "frame_id": frame_id,
                    "snapshot_timestamp_ns": timestamp_ns,
                    "accepted_observations": int(snapshot["accepted_observation_count"]),
                    "rejected_observations": len(snapshot["rejected_observations"]),
                    "association_count": int(snapshot["association_count"]),
                    "multi_source_associations": multi_source,
                    "track_count": len(snapshot["tracks"]),
                    "ego_poses": [
                        _ego_pose_document(
                            ue_id="ue-a", source=source_a, prepared_sensor=prepared_a
                        ),
                        _ego_pose_document(
                            ue_id="ue-b", source=source_b, prepared_sensor=prepared_b
                        ),
                    ],
                    "source_observations": source_observations,
                    "associations": snapshot["associations"],
                    "tracks": snapshot["tracks"],
                })
                fault_case = deterministic_fault_case(fault_case_cursor)
                fault_row = _shadow_case(
                    case=fault_case, payload_a=payload_a, payload_b=payload_b,
                    timestamp_ns=timestamp_ns, session_id=f"{session_id}-shadow-{pair_count}",
                )
                if fault_row["applicable"]:
                    fault_counts[fault_case] += 1
                    fault_case_cursor += 1
                _append_json_line(faults, {
                    "pair_index": pair_count, "frame_id": frame_id, **fault_row
                })
                pair_count += 1
        _require(pair_count == int(args.max_pairs),
                 f"only {pair_count}/{args.max_pairs} paired frames completed before timeout")
        _require(set(fault_counts) == set(FAULT_CASES), "not every shadow fault case executed")
        _require(nominal_multi_source > 0,
                 "no real two-source association was observed; scenario did not demonstrate fusion")
        summary = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "scientific_scope": "LIVE_CARLA_TWO_UE_MODEL_TO_MAP_FUNCTIONAL_DEMONSTRATION",
            "not_claimed": [
                "TWO_DEVICE_NETWORK_LATENCY",
                "RADIO_PERFORMANCE",
                "CALIBRATED_POSITION_COVARIANCE_FUSION",
                "AGENT_SELECTED_ACTIONS",
                "UI_QUALIFICATION",
            ],
            "profile": _profile_document(profile),
            "actor_ids": {"ue-a": int(vehicle_a.id), "ue-b": int(vehicle_b.id)},
            "session_id": session_id,
            "clock_domain": CLOCK_DOMAIN,
            "paired_frames": pair_count,
            "observation_batches": pair_count * 2,
            "nominal_associations": nominal_associations,
            "nominal_multi_source_associations": nominal_multi_source,
            "fault_case_counts": dict(sorted(fault_counts.items())),
            "association_policy": _policy().as_dict(),
            "resident_runtime_shared_by_logical_ues": True,
            "scenario_ownership": (
                "SELF_CONTAINED_OWNED_TWO_EGO"
                if owned_scenario is not None
                else "PASSIVE_EXISTING_EGOS"
            ),
            "existing_ego_vehicles_controlled_or_destroyed": False,
            "sensor_contract": {
                "world_tick_hz": WORLD_TICK_HZ,
                "prepared_input_hz": WORLD_TICK_HZ / PREPARE_EVERY_N_TICKS,
                "prepare_every_n_ticks": PREPARE_EVERY_N_TICKS,
                "rgb": {"width": 1280, "height": 720, "horizontal_fov_deg": 120.0},
                "radar": {
                    "points_per_second": 200000,
                    "range_m": 120.0,
                    "horizontal_fov_deg": 120.0,
                    "vertical_fov_deg": 30.0,
                    "window_sweeps": 2,
                    "window_support_ms": 200.0,
                },
            },
            "raw_sensor_frames_retained": 0,
            "visualization_evidence": {
                "ego_pose_records": pair_count * 2,
                "compact_model_object_records": compact_model_records,
                "association_identity_records": nominal_associations,
                "carla_actor_ground_truth_records": 0,
                "raw_rgb_frames": 0,
                "raw_radar_frames": 0,
                "semantic_masks": 0,
            },
            "elapsed_wall_s": time.time() - started_wall,
            "ue_operation_counters": asdict(ue_runtime.counters),
            "edge_operation_counters": asdict(edge_runtime.counters),
            "ue_call_ledger": ue_ledger.snapshot(),
            "edge_call_ledger": edge_ledger.snapshot(),
            "registry_audit": asdict(registry.startup_audit),
        }
        _write_json_create_only(output / "SUMMARY.json", summary)
        manifest = _manifest(output, summary)
        _write_json_create_only(output / "manifest.json", manifest)
        with (output / COMPLETE_TERMINAL).open("x", encoding="utf-8") as handle:
            handle.write(f"{COMPLETE_TERMINAL} {manifest['summary_sha256']}\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(COMPLETE_TERMINAL)
        return 0
    except BaseException as exc:
        failure = {
            "schema": f"{SCHEMA}.failure",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_wall_s": time.time() - started_wall,
            "progress": {
                "paired_frames": int(pair_count),
                "requested_paired_frames": int(args.max_pairs),
                "last_considered_frame": int(last_frame),
            },
            "sensor_clock_contract": {
                "world_tick_hz": WORLD_TICK_HZ,
                "world_delta_s": WORLD_DELTA_S,
                "prepare_every_n_ticks": PREPARE_EVERY_N_TICKS,
                "prepared_input_hz": WORLD_TICK_HZ / PREPARE_EVERY_N_TICKS,
            },
            "sensor_diagnostics": {
                "ue-a": _safe_sensor_diagnostics(source_a),
                "ue-b": _safe_sensor_diagnostics(source_b),
            },
        }
        try:
            _write_json_create_only(output / "FAILED.json", failure)
        except (FileExistsError, OSError):
            pass
        raise
    finally:
        if source_b is not None:
            source_b.close()
        if source_a is not None:
            source_a.close()
        if owned_scenario is not None:
            owned_scenario.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=10.0)
    parser.add_argument("--list-vehicles", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--spawn-two-egos", action="store_true")
    parser.add_argument("--execute")
    parser.add_argument("--ue-a-actor-id", type=int, default=-1)
    parser.add_argument("--ue-b-actor-id", type=int, default=-1)
    parser.add_argument("--session-id", default="two-ue-live-action50-v1")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--sensor-timeout-s", type=float, default=5.0)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--scenario-seed", type=int, default=31)
    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--ego-spawn-index", type=int, default=80)
    parser.add_argument("--ego-gap-m", type=float, default=15.0)
    parser.add_argument("--fixed-route-spawn-indices", default="80,85,91,94,99,80")
    parser.add_argument("--npc-vehicles", type=int, default=28)
    parser.add_argument("--npc-pedestrians", type=int, default=35)
    parser.add_argument("--spawn-radius-m", type=float, default=80.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.list_vehicles and not args.preflight:
        if not args.output:
            parser.error("--output is required for execution")
        if not args.spawn_two_egos and (args.ue_a_actor_id < 0 or args.ue_b_actor_id < 0):
            parser.error("--ue-a-actor-id and --ue-b-actor-id are required")
        if args.spawn_two_egos and (args.ue_a_actor_id >= 0 or args.ue_b_actor_id >= 0):
            parser.error("--spawn-two-egos cannot be combined with actor IDs")
        if (
            args.max_pairs < 1
            or args.duration_s <= 0.0
            or args.frame_stride < 1
            or args.sensor_timeout_s <= 0.0
            or args.ego_gap_m <= 0.0
            or args.npc_vehicles < 0
            or args.npc_pedestrians < 0
            or args.spawn_radius_m <= 0.0
        ):
            parser.error("execution counts, rates, distances, and timeouts are invalid")
    return args


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

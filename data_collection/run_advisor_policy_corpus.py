#!/usr/bin/env python3
"""Collect the advisor-rich policy corpus with one CARLA sync ticker.

The advisor scripts remain read-only population clients.  This runner owns the
episode lifecycle: it makes the empty world synchronous, ticks while the
populators start, yields sole tick ownership to the fusion collector, ticks
their shutdown, verifies cleanup, and restores asynchronous mode.  The
collector is always launched in observe-existing mode by the resolved config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import carla
import numpy as np
import pandas as pd
import yaml

from data_collection import run_policy_corpus as base_runner
from rl_agent.policy.replay import _greedy_prediction_matches, _normalize_class


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "policy_corpus_advisor_rich_v5.yaml"
)
DYNAMIC_PATTERNS = (
    "vehicle.*",
    "walker.pedestrian.*",
    "sensor.*",
    "controller.ai.walker",
)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _connect(config: Mapping[str, object]) -> Tuple[carla.Client, carla.World]:
    connection = config["carla"]
    last_error = ""
    # The packaged renderer can reject RPC handshakes briefly after the static
    # preflight's GPU inventory query. Direct probes recover immediately after
    # the old 10 s window, so keep the retry bounded but long enough to cross
    # that observed startup/backpressure interval.
    attempts = 30
    for _attempt in range(attempts):
        try:
            client = carla.Client(str(connection["host"]), int(connection["port"]))
            client.set_timeout(float(connection.get("timeout_s", 10.0)))
            # This CARLA 0.10 Linux package intermittently aborts get_world()
            # when it is the first RPC on a fresh client. The lightweight
            # version request reliably establishes the session and is needed
            # for the required version check anyway.
            server_version = str(client.get_server_version())
            world = client.get_world()
            if not str(world.get_map().name).endswith(str(connection["expected_town"])):
                raise RuntimeError(
                    f"expected {connection['expected_town']}, found {world.get_map().name}"
                )
            if server_version != str(connection["expected_server_version"]):
                raise RuntimeError(
                    f"expected CARLA {connection['expected_server_version']}, "
                    f"found {server_version}"
                )
            return client, world
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(
        f"CARLA connection failed after {attempts} attempts: {last_error}"
    )


def _actor_inventory(world: carla.World) -> Dict[str, int]:
    actors = world.get_actors()
    return {pattern: int(len(actors.filter(pattern))) for pattern in DYNAMIC_PATTERNS}


def _role_inventory(world: carla.World) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for actor in world.get_actors():
        try:
            role = str(actor.attributes.get("role_name", "")).strip()
        except (AttributeError, RuntimeError):
            continue
        if role:
            counts[role] = counts.get(role, 0) + 1
    return counts


def _matching_role_count(roles: Mapping[str, int], prefix: str) -> int:
    return sum(count for role, count in roles.items() if role.startswith(prefix))


def _longest_mask_dwell_s(mask: pd.Series, timestamps: pd.Series) -> float:
    if mask.empty:
        return 0.0
    return _longest_true_dwell(
        mask.astype(bool).tolist(),
        pd.to_numeric(timestamps, errors="coerce").tolist(),
    )


def _traffic_sanity_summary(
    trajectories: pd.DataFrame,
    collisions: pd.DataFrame,
    expected_actor_ids: Sequence[int],
    gate: Mapping[str, object],
) -> Dict[str, object]:
    """Summarize NPC collisions and sustained network-wide gridlock."""

    expected_ids = {int(value) for value in expected_actor_ids}
    if not expected_ids:
        return {
            "applicable": False,
            "pass": True,
            "failures": [],
            "monitored_npc_vehicles": 0,
            "collision_events": 0,
            "persistent_gridlock_dwell_s": 0.0,
        }

    trajectories = trajectories.copy()
    collisions = collisions.copy()
    observed_ids = set(
        pd.to_numeric(trajectories.get("actor_id", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
    )
    observation_fraction = len(expected_ids & observed_ids) / len(expected_ids)

    collision_incidents = 0
    if not collisions.empty:
        collision_rows = collisions.copy()
        collision_rows["frame_id"] = pd.to_numeric(
            collision_rows["frame_id"], errors="coerce"
        ).fillna(-1).astype(int)
        first = pd.to_numeric(collision_rows["npc_actor_id"], errors="coerce").fillna(-1).astype(int)
        second = pd.to_numeric(collision_rows["other_actor_id"], errors="coerce").fillna(-1).astype(int)
        collision_rows["pair_low"] = np.minimum(first, second)
        collision_rows["pair_high"] = np.maximum(first, second)
        collision_rows = collision_rows.sort_values(
            ["pair_low", "pair_high", "frame_id"]
        )
        previous_frame = collision_rows.groupby(["pair_low", "pair_high"])[
            "frame_id"
        ].shift()
        collision_incidents = int(
            (previous_frame.isna() | ((collision_rows["frame_id"] - previous_frame) > 2)).sum()
        )

    persistent_gridlock_dwell_s = 0.0
    npc_speed_p50_mps = None
    stopped_fraction_p95 = None
    if not trajectories.empty:
        trajectories["speed_mps"] = pd.to_numeric(
            trajectories["speed_mps"], errors="coerce"
        )
        valid_speed = trajectories["speed_mps"].dropna()
        if len(valid_speed):
            npc_speed_p50_mps = float(valid_speed.quantile(0.50))
        per_frame = trajectories.groupby("frame_id").agg(
            carla_timestamp=("carla_timestamp", "median"),
            npc_count=("actor_id", "nunique"),
            stopped_fraction=(
                "speed_mps",
                lambda values: float(
                    (values <= float(gate["stopped_speed_max_mps"])).mean()
                ),
            ),
        ).sort_index()
        if len(per_frame):
            stopped_fraction_p95 = float(per_frame["stopped_fraction"].quantile(0.95))
            gridlocked = (
                per_frame["npc_count"] >= int(gate["gridlock_minimum_npc_count"])
            ) & (
                per_frame["stopped_fraction"]
                >= float(gate["gridlock_stopped_fraction"])
            )
            persistent_gridlock_dwell_s = _longest_mask_dwell_s(
                gridlocked, per_frame["carla_timestamp"]
            )

    failures: List[str] = []
    if observation_fraction < float(gate["minimum_actor_observation_fraction"]):
        failures.append("insufficient_npc_trajectory_observation")
    if collision_incidents > int(gate["maximum_collision_incidents"]):
        failures.append("npc_collision_incidents_above_gate")
    if persistent_gridlock_dwell_s >= float(gate["persistent_gridlock_min_s"]):
        failures.append("persistent_network_gridlock")
    return {
        "applicable": True,
        "pass": not failures,
        "failures": failures,
        "monitored_npc_vehicles": int(len(expected_ids)),
        "observed_npc_vehicles": int(len(expected_ids & observed_ids)),
        "actor_observation_fraction": float(observation_fraction),
        "collision_callback_rows": int(len(collisions)),
        "collision_events": int(collision_incidents),
        "persistent_gridlock_dwell_s": float(persistent_gridlock_dwell_s),
        "npc_speed_p50_mps": npc_speed_p50_mps,
        "stopped_fraction_p95": stopped_fraction_p95,
    }


def _radar_density_summary(
    metrics: pd.DataFrame, gate: Mapping[str, object]
) -> Dict[str, object]:
    """Gate the realized radar evidence, not only requested blueprint args."""

    values = pd.to_numeric(
        metrics.get("radar_projected_points", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    reference = float(gate["reference_projected_points_median"])
    tolerance = float(gate["relative_tolerance"])
    lower = reference * (1.0 - tolerance)
    upper = reference * (1.0 + tolerance)
    median = float(values.median()) if len(values) else None
    failures: List[str] = []
    if len(values) < int(gate["minimum_metric_frames"]):
        failures.append("insufficient_radar_density_frames")
    if median is None:
        failures.append("missing_radar_projected_points")
    elif not lower <= median <= upper:
        failures.append("radar_projected_points_median_outside_contract")
    return {
        "pass": not failures,
        "failures": failures,
        "frames": int(len(values)),
        "reference_projected_points_median": reference,
        "relative_tolerance": tolerance,
        "minimum_projected_points_median": lower,
        "maximum_projected_points_median": upper,
        "radar_projected_points_median": median,
        "median_fraction_of_reference": (
            median / reference if median is not None and reference > 0.0 else None
        ),
        "radar_projected_points_p05": (
            float(values.quantile(0.05)) if len(values) else None
        ),
        "radar_projected_points_p95": (
            float(values.quantile(0.95)) if len(values) else None
        ),
    }


def _exact_fast_scenario_summary(
    ground_truth: pd.DataFrame,
    route: pd.DataFrame,
    *,
    role_name: str,
    maximum_route_offset_m: float,
    pedestrian_speed_max_mps: float,
) -> Dict[str, object]:
    """Reject off-route exact convoys and lead/walker impact signatures."""

    exact = ground_truth[ground_truth["role_name"].astype(str) == role_name].copy()
    failures: List[str] = []
    maximum_offset = None
    if exact.empty:
        failures.append("missing_exact_fast_target_gt")
    else:
        route_xy = route[["ego_x", "ego_y"]].to_numpy(dtype=float)
        exact_xy = exact[["origin_x", "origin_y"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(exact_xy).all(axis=1)
        offsets = [
            float(np.hypot(route_xy[:, 0] - x, route_xy[:, 1] - y).min())
            for x, y in exact_xy[finite]
        ]
        maximum_offset = max(offsets) if offsets else None
        if maximum_offset is None or maximum_offset > maximum_route_offset_m:
            failures.append("exact_fast_target_left_authored_route")

    pedestrian = ground_truth[
        ground_truth["class_name"].map(_normalize_class) == "pedestrian"
    ].copy()
    pedestrian_speed_parts = []
    for _actor_id, group in pedestrian.groupby("actor_id"):
        group = group.sort_values("carla_timestamp")
        timestamp = pd.to_numeric(group["carla_timestamp"], errors="coerce")
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        dt = timestamp.diff()
        pedestrian_speed_parts.append(pd.Series(np.hypot(dx, dy) / dt))
    pedestrian_speed = (
        pd.concat(pedestrian_speed_parts, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        if pedestrian_speed_parts
        else pd.Series(dtype=float)
    )
    pedestrian_speed_max = (
        float(pedestrian_speed.max()) if len(pedestrian_speed) else None
    )
    pedestrian_speed_rows_above_max = int(
        (pedestrian_speed > pedestrian_speed_max_mps).sum()
    )
    if pedestrian_speed_rows_above_max:
        failures.append("exact_fast_pedestrian_impact_signature")
    return {
        "applicable": True,
        "pass": not failures,
        "failures": failures,
        "exact_fast_target_rows": int(len(exact)),
        "maximum_route_offset_m": maximum_offset,
        "maximum_route_offset_gate_m": float(maximum_route_offset_m),
        "pedestrian_rows": int(len(pedestrian)),
        "pedestrian_speed_max_mps": pedestrian_speed_max,
        "pedestrian_speed_gate_mps": float(pedestrian_speed_max_mps),
        "pedestrian_speed_rows_above_max": pedestrian_speed_rows_above_max,
    }


def _pedestrian_motion_summary(
    ground_truth: pd.DataFrame,
    *,
    maximum_speed_mps: float,
) -> Dict[str, object]:
    """Detect walker impact/push signatures in every scenario family."""

    pedestrian = ground_truth[
        ground_truth["class_name"].map(_normalize_class) == "pedestrian"
    ].copy()
    speed_parts = []
    for _actor_id, group in pedestrian.groupby("actor_id"):
        group = group.sort_values("carla_timestamp")
        timestamp = pd.to_numeric(group["carla_timestamp"], errors="coerce")
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        dt = timestamp.diff()
        speed_parts.append(pd.Series(np.hypot(dx, dy) / dt))
    speeds = (
        pd.concat(speed_parts, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        if speed_parts else pd.Series(dtype=float)
    )
    rows_above = int((speeds > float(maximum_speed_mps)).sum())
    return {
        "pass": rows_above == 0,
        "failures": ["pedestrian_impact_signature"] if rows_above else [],
        "pedestrian_rows": int(len(pedestrian)),
        "pedestrian_speed_max_mps": float(speeds.max()) if len(speeds) else None,
        "pedestrian_speed_gate_mps": float(maximum_speed_mps),
        "pedestrian_speed_rows_above_max": rows_above,
    }


def _in_forward_corridor(
    transform: object,
    other_location: object,
    *,
    maximum_forward_m: float,
    maximum_lateral_m: float,
) -> bool:
    location = transform.location
    forward = transform.get_forward_vector()
    dx = float(other_location.x) - float(location.x)
    dy = float(other_location.y) - float(location.y)
    forward_m = dx * float(forward.x) + dy * float(forward.y)
    lateral_m = abs(-dx * float(forward.y) + dy * float(forward.x))
    return bool(
        0.0 < forward_m <= float(maximum_forward_m)
        and lateral_m <= float(maximum_lateral_m)
    )


def _walker_requires_yield(
    transform: object,
    walker_location: object,
    walker_velocity: object,
) -> bool:
    speed_mps = math.hypot(float(walker_velocity.x), float(walker_velocity.y))
    lateral_limit = 3.5 if speed_mps >= 0.2 else 2.2
    return _in_forward_corridor(
        transform,
        walker_location,
        maximum_forward_m=15.0,
        maximum_lateral_m=lateral_limit,
    )


def _vehicle_requires_yield(
    transform: object,
    vehicle_location: object,
    *,
    maximum_forward_m: float = 12.0,
) -> bool:
    """Use a widening forward corridor for vehicles on curved loop segments."""

    location = transform.location
    forward = transform.get_forward_vector()
    dx = float(vehicle_location.x) - float(location.x)
    dy = float(vehicle_location.y) - float(location.y)
    forward_m = dx * float(forward.x) + dy * float(forward.y)
    lateral_m = abs(-dx * float(forward.y) + dy * float(forward.x))
    lateral_limit_m = min(6.0, 2.8 + 0.35 * max(0.0, forward_m))
    return bool(
        0.0 < forward_m <= float(maximum_forward_m)
        and lateral_m <= lateral_limit_m
    )


class TrafficSanityMonitor:
    """Passive trajectory logger plus per-NPC CARLA collision sensors."""

    def __init__(
        self,
        *,
        world: carla.World,
        traffic_manager: object,
        output_dir: Path,
        integration: Mapping[str, object],
    ) -> None:
        self.world = world
        self.traffic_manager = traffic_manager
        self.output_dir = output_dir
        self.integration = integration
        self.actor_ids: List[int] = []
        self.actor_metadata: Dict[int, Dict[str, object]] = {}
        self.collision_sensors: List[carla.Actor] = []
        self.trajectory_rows: List[Dict[str, object]] = []
        self.collision_rows: List[Dict[str, object]] = []
        self._tick_callback_id = None
        self._lock = threading.Lock()
        self.initial_geometry: Dict[str, object] = {}
        self._direct_route_state: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _actor_role(actor: carla.Actor) -> str:
        try:
            return str(actor.attributes.get("role_name", ""))
        except (AttributeError, RuntimeError):
            return ""

    def _npc_vehicles(self) -> List[carla.Actor]:
        vehicles = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            role = self._actor_role(actor)
            if role.startswith("autopilot") or role == "hero":
                vehicles.append(actor)
        return vehicles

    def _npc_loop_route(self) -> List[carla.Location]:
        path = base_runner._resolve_repo_path(
            str(self.integration["npc_loop_route_progress_csv"])
        )
        locations: List[carla.Location] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                locations.append(
                    carla.Location(
                        x=float(row["ego_x"]),
                        y=float(row["ego_y"]),
                        z=float(row.get("ego_z", 0.0)),
                    )
                )
        if len(locations) < 2:
            raise RuntimeError(f"NPC loop route is invalid: {path}")
        return locations

    @staticmethod
    def _rotated_loop_path(
        actor: carla.Actor,
        route: Sequence[carla.Location],
        repetitions: int,
    ) -> List[carla.Location]:
        actor_location = actor.get_location()
        index = min(
            range(len(route)),
            key=lambda route_index: actor_location.distance(route[route_index]),
        )
        one_loop = [*route[index + 1 :], *route[: index + 1]]
        return one_loop * max(1, int(repetitions))

    @staticmethod
    def _minimum_pairwise_distance(actors: Sequence[carla.Actor]) -> float | None:
        locations = [actor.get_location() for actor in actors]
        if len(locations) < 2:
            return None
        return float(
            min(
                locations[left].distance(locations[right])
                for left in range(len(locations))
                for right in range(left + 1, len(locations))
            )
        )

    def start(self) -> None:
        vehicles = self._npc_vehicles()
        blockers = [
            actor
            for actor in self.world.get_actors().filter("vehicle.*")
            if self._actor_role(actor).startswith("static_blocker_v4")
        ]
        self.actor_ids = [int(actor.id) for actor in vehicles]
        self.actor_metadata = {
            int(actor.id): {
                "role_name": self._actor_role(actor),
                "type_id": str(actor.type_id),
            }
            for actor in vehicles
        }
        leading_distance = float(self.integration["tm_distance_to_leading_vehicle_m"])
        speed_difference = float(self.integration["tm_speed_difference_pct"])
        desired_speed = float(self.integration["tm_desired_speed_mps"])
        route_mode = str(self.integration.get("npc_route_mode", "fixed_loop"))
        if route_mode not in {"fixed_loop", "tm_autonomous", "direct_loop"}:
            raise ValueError(f"unsupported NPC route mode: {route_mode}")
        loop_route = (
            self._npc_loop_route()
            if route_mode in {"fixed_loop", "direct_loop"}
            else []
        )
        loop_repetitions = int(self.integration.get("npc_loop_repetitions", 1))
        self.traffic_manager.set_global_distance_to_leading_vehicle(leading_distance)
        self.traffic_manager.global_percentage_speed_difference(speed_difference)
        tm_failures = []
        for actor in vehicles:
            try:
                self.traffic_manager.distance_to_leading_vehicle(actor, leading_distance)
                self.traffic_manager.vehicle_percentage_speed_difference(
                    actor, speed_difference
                )
                self.traffic_manager.set_desired_speed(actor, desired_speed)
                self.traffic_manager.auto_lane_change(actor, False)
                if route_mode == "fixed_loop":
                    self.traffic_manager.set_path(
                        actor,
                        self._rotated_loop_path(actor, loop_route, loop_repetitions),
                    )
                elif route_mode == "direct_loop":
                    actor.set_autopilot(False, int(self.integration["tm_port"]))
                    actor_location = actor.get_location()
                    route_index = min(
                        range(len(loop_route)),
                        key=lambda index: actor_location.distance(loop_route[index]),
                    )
                    self._direct_route_state[int(actor.id)] = {
                        "waypoint_index": int(route_index),
                        "route": loop_route,
                    }
                for blocker in blockers:
                    self.traffic_manager.collision_detection(actor, blocker, True)
                for other in vehicles:
                    if int(other.id) != int(actor.id):
                        self.traffic_manager.collision_detection(actor, other, True)
            except RuntimeError as exc:
                tm_failures.append({"actor_id": int(actor.id), "error": str(exc)})
        if tm_failures:
            raise RuntimeError(f"could not apply NPC Traffic Manager safety profile: {tm_failures}")

        minimum_blocker_distance = None
        if vehicles and blockers:
            minimum_blocker_distance = float(
                min(
                    vehicle.get_location().distance(blocker.get_location())
                    for vehicle in vehicles
                    for blocker in blockers
                )
            )
        self.initial_geometry = {
            "npc_vehicle_count": int(len(vehicles)),
            "static_blocker_count": int(len(blockers)),
            "minimum_npc_pairwise_distance_m": self._minimum_pairwise_distance(vehicles),
            "minimum_npc_to_static_blocker_distance_m": minimum_blocker_distance,
            "tm_distance_to_leading_vehicle_m": leading_distance,
            "tm_speed_difference_pct": speed_difference,
            "tm_desired_speed_mps": desired_speed,
            "npc_route_mode": route_mode,
            "npc_loop_route_points": int(len(loop_route)),
            "npc_loop_repetitions": (
                int(loop_repetitions)
                if route_mode in {"fixed_loop", "direct_loop"}
                else None
            ),
            "safe_blueprint_filter_enabled": True,
        }

        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        for actor in vehicles:
            sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=actor)
            sensor.listen(
                lambda event, npc_id=int(actor.id): self._on_collision(npc_id, event)
            )
            self.collision_sensors.append(sensor)
        self._tick_callback_id = self.world.on_tick(self._on_tick)

    def _on_tick(self, snapshot: object) -> None:
        rows = []
        timestamp = float(snapshot.timestamp.elapsed_seconds)
        if self._direct_route_state:
            self._apply_direct_route_controls()
        for actor_id in self.actor_ids:
            actor_snapshot = snapshot.find(int(actor_id))
            if actor_snapshot is None:
                continue
            transform = actor_snapshot.get_transform()
            velocity = actor_snapshot.get_velocity()
            metadata = self.actor_metadata[actor_id]
            rows.append(
                {
                    "frame_id": int(snapshot.frame),
                    "carla_timestamp": timestamp,
                    "actor_id": int(actor_id),
                    "role_name": metadata["role_name"],
                    "type_id": metadata["type_id"],
                    "world_x": float(transform.location.x),
                    "world_y": float(transform.location.y),
                    "world_z": float(transform.location.z),
                    "speed_mps": float(
                        math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    ),
                }
            )
        with self._lock:
            self.trajectory_rows.extend(rows)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _apply_direct_route_controls(self) -> None:
        """Apply bounded synchronous waypoint control to all managed NPCs."""

        actors = {
            int(actor.id): actor
            for actor in self.world.get_actors(self.actor_ids)
            if actor is not None
        }
        all_vehicles = {
            int(actor.id): actor
            for actor in self.world.get_actors().filter("vehicle.*")
        }
        all_walkers = list(self.world.get_actors().filter("walker.pedestrian.*"))
        target_speed = float(self.integration["npc_direct_route_speed_mps"])
        for actor_id, state in self._direct_route_state.items():
            actor = actors.get(actor_id)
            if actor is None:
                continue
            route = state["route"]
            index = int(state["waypoint_index"])
            transform = actor.get_transform()
            location = transform.location
            for _unused in range(len(route)):
                target = route[index]
                if location.distance(target) >= 4.0:
                    break
                index = (index + 1) % len(route)
            lookahead = route[(index + 1) % len(route)]
            desired_yaw = math.atan2(
                float(lookahead.y) - float(location.y),
                float(lookahead.x) - float(location.x),
            )
            heading_error = self._wrap_angle(
                desired_yaw - math.radians(float(transform.rotation.yaw))
            )
            velocity = actor.get_velocity()
            speed_mps = math.sqrt(
                float(velocity.x) ** 2
                + float(velocity.y) ** 2
                + float(velocity.z) ** 2
            )
            turn_scale = max(0.30, 1.0 - abs(heading_error) / math.pi)
            speed_error = target_speed * turn_scale - speed_mps
            throttle = max(0.0, min(0.65, 0.30 * speed_error))
            brake = max(0.0, min(0.75, -0.40 * speed_error))
            steer = max(-0.70, min(0.70, heading_error / math.radians(45.0)))

            for other_id, other in all_vehicles.items():
                if other_id == actor_id:
                    continue
                other_location = other.get_location()
                if _vehicle_requires_yield(
                    transform,
                    other_location,
                    maximum_forward_m=12.0,
                ):
                    throttle = 0.0
                    brake = 1.0
                    break
            if brake < 1.0:
                for walker in all_walkers:
                    try:
                        walker_location = walker.get_location()
                        walker_velocity = walker.get_velocity()
                    except RuntimeError:
                        continue
                    if _walker_requires_yield(
                        transform,
                        walker_location,
                        walker_velocity,
                    ):
                        throttle = 0.0
                        brake = 1.0
                        break
            actor.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake),
                    hand_brake=False,
                )
            )
            state["waypoint_index"] = int(index)

    def _on_collision(self, npc_actor_id: int, event: object) -> None:
        other = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        row = {
            "frame_id": int(getattr(event, "frame", -1)),
            "carla_timestamp": float(getattr(event, "timestamp", float("nan"))),
            "npc_actor_id": int(npc_actor_id),
            "npc_role_name": self.actor_metadata.get(npc_actor_id, {}).get("role_name", ""),
            "other_actor_id": int(getattr(other, "id", -1)),
            "other_type_id": str(getattr(other, "type_id", "")),
            "other_role_name": self._actor_role(other) if other is not None else "",
            "normal_impulse_x": float(getattr(impulse, "x", 0.0)),
            "normal_impulse_y": float(getattr(impulse, "y", 0.0)),
            "normal_impulse_z": float(getattr(impulse, "z", 0.0)),
        }
        with self._lock:
            self.collision_rows.append(row)

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)

    def stop(self) -> Dict[str, object]:
        if self._tick_callback_id is not None:
            try:
                self.world.remove_on_tick(self._tick_callback_id)
            except RuntimeError:
                pass
            self._tick_callback_id = None
        for sensor in reversed(self.collision_sensors):
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass
        self.collision_sensors.clear()
        with self._lock:
            trajectory_rows = list(self.trajectory_rows)
            collision_rows = list(self.collision_rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_fields = (
            "frame_id", "carla_timestamp", "actor_id", "role_name", "type_id",
            "world_x", "world_y", "world_z", "speed_mps",
        )
        collision_fields = (
            "frame_id", "carla_timestamp", "npc_actor_id", "npc_role_name",
            "other_actor_id", "other_type_id", "other_role_name",
            "normal_impulse_x", "normal_impulse_y", "normal_impulse_z",
        )
        self._write_csv(self.output_dir / "npc_trajectories.csv", trajectory_rows, trajectory_fields)
        self._write_csv(self.output_dir / "npc_collision_events.csv", collision_rows, collision_fields)
        summary = _traffic_sanity_summary(
            pd.DataFrame(trajectory_rows, columns=trajectory_fields),
            pd.DataFrame(collision_rows, columns=collision_fields),
            self.actor_ids,
            self.integration["traffic_sanity_gate"],
        )
        summary["initial_geometry"] = self.initial_geometry
        (self.output_dir / "traffic_sanity_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary


def _require_empty_async(world: carla.World) -> Dict[str, object]:
    inventory = _actor_inventory(world)
    occupied = {name: value for name, value in inventory.items() if value}
    settings = world.get_settings()
    if occupied:
        raise RuntimeError(f"advisor corpus requires an empty dynamic world: {occupied}")
    if bool(settings.synchronous_mode):
        raise RuntimeError(
            "advisor corpus requires asynchronous startup; a stale synchronous world has no known owner"
        )
    return {
        "dynamic_actor_counts": inventory,
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": settings.fixed_delta_seconds,
    }


def _set_sync_master(
    client: carla.Client,
    world: carla.World,
    tm_port: int,
    fixed_delta_seconds: float,
) -> object:
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(fixed_delta_seconds)
    world.apply_settings(settings)
    client.get_trafficmanager(int(tm_port)).set_synchronous_mode(True)
    world.tick(2.0)
    return original


def _restore_async(
    client: carla.Client,
    world: carla.World,
    tm_port: int,
    original_settings: object,
) -> None:
    client.get_trafficmanager(int(tm_port)).set_synchronous_mode(False)
    world.apply_settings(original_settings)


def _spawn_ego_reservations(
    world: carla.World,
    spawn_indices: Sequence[int],
) -> List[carla.Actor]:
    spawn_points = list(world.get_map().get_spawn_points())
    actors: List[carla.Actor] = []
    try:
        for spawn_index in spawn_indices:
            if not 0 <= int(spawn_index) < len(spawn_points):
                raise ValueError(f"ego reservation spawn index {spawn_index} is invalid")
            blueprint = world.get_blueprint_library().find("vehicle.lincoln.mkz")
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute(
                    "role_name", f"advisor_ego_spawn_reservation_{int(spawn_index)}"
                )
            actor = world.try_spawn_actor(blueprint, spawn_points[int(spawn_index)])
            if actor is None:
                raise RuntimeError(
                    f"unable to reserve advisor route-corridor spawn {spawn_index}"
                )
            actor.set_simulate_physics(False)
            actors.append(actor)
        return actors
    except Exception:
        _destroy_ego_reservations(world, actors)
        raise


def _destroy_ego_reservations(
    world: carla.World, actors: Sequence[carla.Actor]
) -> None:
    if not actors:
        return
    for actor in reversed(list(actors)):
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass
    world.tick(2.0)


def _population_commands(
    config: Mapping[str, object], run_spec: Mapping[str, object]
) -> Tuple[List[str], List[str]]:
    integration = config["advisor_integration"]
    family = str(run_spec["scenario_family"])
    family_spec = integration["families"][family]
    host = str(config["carla"]["host"])
    port = str(config["carla"]["port"])
    tm_port = str(integration["tm_port"])
    blocker_overrides = [
        *(str(value) for value in integration.get("common_blocker_args", [])),
        *(str(value) for value in family_spec.get("blocker_args", [])),
    ]
    pedestrian_blockers_enabled = "--no-pedestrian-blockers" not in blocker_overrides
    pedestrian_location_args = [
        str(value)
        for location in integration["pedestrian_locations"]
        for value in ("--pedestrian-location", *location)
    ] if pedestrian_blockers_enabled else []
    blocker = [
        sys.executable,
        "-u",
        str(base_runner._resolve_repo_path(str(integration["spawn_blocker_entrypoint"]))),
        "--host",
        host,
        "--port",
        port,
        "--ego-role-name",
        str(integration["ego_role_name"]),
        "--pedestrian-speed",
        str(integration["pedestrian_speed_mps"]),
        "--min-pedestrian-speed",
        str(integration["minimum_pedestrian_speed_mps"]),
        "--update-hz",
        str(integration["update_hz"]),
        "--tick-timeout",
        str(integration["tick_timeout_s"]),
        "--no-intercept-debug",
        *pedestrian_location_args,
        *blocker_overrides,
    ]
    if (
        "--no-pedestrian-blockers" in blocker_overrides
        and "--no-vehicle-blockers" in blocker_overrides
    ):
        blocker = []
    traffic = [
        sys.executable,
        "-u",
        str(
            base_runner._resolve_repo_path(
                str(
                    integration.get(
                        "generate_traffic_entrypoint",
                        integration["generate_traffic_script"],
                    )
                )
            )
        ),
        "--host",
        host,
        "--port",
        port,
        "--tm-port",
        tm_port,
        "--number-of-vehicles",
        str(family_spec["number_of_vehicles"]),
        "--number-of-walkers",
        str(family_spec["number_of_walkers"]),
        "--seed",
        str(run_spec["seed"]),
        "--seedw",
        str(int(run_spec["seed"]) + int(integration["walker_seed_offset"])),
        "--replenish-interval",
        str(integration["replenish_interval_s"]),
        "--population-log-interval",
        str(integration["population_log_interval_s"]),
        *(str(value) for value in integration.get("common_traffic_args", [])),
        *(str(value) for value in family_spec.get("traffic_args", [])),
    ]
    return blocker, traffic


def _start_process(command: Sequence[str], log_path: Path) -> Tuple[subprocess.Popen, object]:
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def _tick_until(
    world: carla.World,
    processes: Sequence[subprocess.Popen],
    predicate,
    timeout_s: float,
    label: str,
) -> Dict[str, object]:
    deadline = time.monotonic() + float(timeout_s)
    last_inventory: Dict[str, int] = {}
    last_roles: Dict[str, int] = {}
    while time.monotonic() < deadline:
        failures = [process.returncode for process in processes if process.poll() is not None]
        if failures:
            raise RuntimeError(f"{label} process exited early: returncodes={failures}")
        world.tick(2.0)
        last_inventory = _actor_inventory(world)
        last_roles = _role_inventory(world)
        if predicate(last_inventory, last_roles):
            return {"actor_counts": last_inventory, "role_counts": last_roles}
        time.sleep(0.01)
    raise RuntimeError(
        f"timed out waiting for {label}; actor_counts={last_inventory}, roles={last_roles}"
    )


def _blocker_ready(
    inventory: Mapping[str, int],
    roles: Mapping[str, int],
    family_spec: Mapping[str, object],
) -> bool:
    del inventory
    required = family_spec.get("minimum_blocker_role_prefix_counts", {})
    return all(
        _matching_role_count(roles, str(prefix)) >= int(count)
        for prefix, count in required.items()
    )


def _population_ready(
    inventory: Mapping[str, int],
    roles: Mapping[str, int],
    family_spec: Mapping[str, object],
) -> bool:
    required = family_spec.get("minimum_ready_actor_counts", {})
    if not all(int(inventory.get(str(pattern), 0)) >= int(count) for pattern, count in required.items()):
        return False
    minimum_autopilot = int(family_spec.get("minimum_autopilot_vehicles", 0))
    return _matching_role_count(roles, "autopilot") >= minimum_autopilot


def _stop_processes(
    world: carla.World,
    processes: Sequence[Tuple[str, subprocess.Popen, object]],
    timeout_s: float,
) -> List[Dict[str, object]]:
    for _name, process, _stream in reversed(processes):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline and any(
        process.poll() is None for _name, process, _stream in processes
    ):
        try:
            world.tick(2.0)
        except RuntimeError:
            pass
        time.sleep(0.02)
    for _name, process, _stream in reversed(processes):
        if process.poll() is None:
            process.terminate()
    terminate_deadline = time.monotonic() + 3.0
    while time.monotonic() < terminate_deadline and any(
        process.poll() is None for _name, process, _stream in processes
    ):
        try:
            world.tick(2.0)
        except RuntimeError:
            pass
        time.sleep(0.02)
    results: List[Dict[str, object]] = []
    for name, process, stream in processes:
        if process.poll() is None:
            process.kill()
        returncode = process.wait(timeout=3.0)
        stream.close()
        results.append({"name": name, "returncode": int(returncode)})
    return results


def _tick_until_empty(world: carla.World, timeout_s: float) -> Dict[str, int]:
    """Wait for populator cleanup without advancing a stale sync world.

    A leaked actor can make ``World.tick`` block indefinitely during teardown,
    which previously left a failed corpus runner alive after both populators
    had exited.  Actor destruction is an RPC and does not require a new frame,
    so poll the inventory passively.  If the ownership-scoped clients still
    leak actors, destroy only the exact remaining dynamic actors as recovery
    and fail the episode so recovered state is never accepted as clean data.
    """

    deadline = time.monotonic() + float(timeout_s)
    inventory = _actor_inventory(world)
    while time.monotonic() < deadline:
        if not any(inventory.values()):
            return inventory
        time.sleep(0.02)
        inventory = _actor_inventory(world)

    leaked_actors: Dict[int, carla.Actor] = {}
    for pattern in DYNAMIC_PATTERNS:
        for actor in world.get_actors().filter(pattern):
            leaked_actors[int(actor.id)] = actor
    leaked_descriptions = [
        {
            "actor_id": int(actor.id),
            "type_id": str(actor.type_id),
            "role_name": str(actor.attributes.get("role_name", "")),
        }
        for actor in leaked_actors.values()
    ]
    for actor in reversed(list(leaked_actors.values())):
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except (AttributeError, RuntimeError):
            pass
        try:
            if actor.is_alive:
                actor.destroy()
        except (AttributeError, RuntimeError):
            pass

    recovery_deadline = time.monotonic() + min(3.0, max(0.5, float(timeout_s)))
    recovery_inventory = _actor_inventory(world)
    while time.monotonic() < recovery_deadline and any(recovery_inventory.values()):
        time.sleep(0.02)
        recovery_inventory = _actor_inventory(world)
    raise RuntimeError(
        "dynamic actors leaked after advisor episode; exact-ID recovery was "
        f"attempted: leaked={leaked_descriptions}, remaining={recovery_inventory}"
    )


def _static_preflight(config: Mapping[str, object]) -> Dict[str, object]:
    preflight = base_runner._static_preflight(config)
    integration = config["advisor_integration"]
    files = {
        "generate_traffic": base_runner._resolve_repo_path(
            str(integration["generate_traffic_script"])
        ),
        "spawn_blocker": base_runner._resolve_repo_path(
            str(integration["spawn_blocker_script"])
        ),
        "spawn_blocker_entrypoint": base_runner._resolve_repo_path(
            str(integration["spawn_blocker_entrypoint"])
        ),
        "route_config": base_runner._resolve_repo_path(str(integration["route_config"])),
        "route_progress_csv": base_runner._resolve_repo_path(
            str(integration["route_progress_csv"])
        ),
        "scenario_ui_v2": base_runner._resolve_repo_path(
            str(integration["scenario_ui_v2"])
        ),
        "scenario_config_v2": base_runner._resolve_repo_path(
            str(integration["scenario_config_v2"])
        ),
        "ego_route_config": base_runner._resolve_repo_path(
            str(integration["ego_route_config_module"])
        ),
        "pole_camera_client": base_runner._resolve_repo_path(
            str(integration["pole_camera_client"])
        ),
        "traffic_light_data": base_runner._resolve_repo_path(
            str(integration["traffic_light_data"])
        ),
    }
    if integration.get("generate_traffic_entrypoint"):
        files["generate_traffic_entrypoint"] = base_runner._resolve_repo_path(
            str(integration["generate_traffic_entrypoint"])
        )
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing advisor integration prerequisites: " + ", ".join(missing))
    route = json.loads(files["route_config"].read_text(encoding="utf-8"))
    if not str(route.get("map", "")).endswith(str(config["carla"]["expected_town"])):
        raise ValueError("advisor route map does not match collection town")
    if route.get("loop") is not True or len(route.get("planned_path", [])) < 2:
        raise ValueError("advisor route must be a non-empty loop")
    progress = pd.read_csv(files["route_progress_csv"])
    if not {"ego_x", "ego_y", "ego_z"}.issubset(progress.columns) or len(progress) < 2:
        raise ValueError("advisor route progress CSV is invalid")
    pedestrian_speed = float(integration["pedestrian_speed_mps"])
    minimum_speed = float(integration["minimum_pedestrian_speed_mps"])
    if not 1.0 <= pedestrian_speed <= 2.0:
        raise ValueError("reactive pedestrian speed must remain in the pinned 1-2 m/s walking band")
    if not 0.0 <= minimum_speed <= pedestrian_speed:
        raise ValueError("minimum pedestrian speed is invalid")
    if int(integration["tm_port"]) != 8010:
        raise ValueError("advisor integration Traffic Manager port must be 8010")
    if "traffic_sanity_gate" in integration:
        if "--safe" not in [str(value) for value in integration.get("common_traffic_args", [])]:
            raise ValueError("advisor traffic generation must enable the --safe blueprint filter")
        if float(integration["tm_distance_to_leading_vehicle_m"]) < 2.5:
            raise ValueError("Traffic Manager following distance must be at least 2.5 m")
        if not 0.0 <= float(integration["tm_speed_difference_pct"]) <= 80.0:
            raise ValueError("Traffic Manager speed difference must be within 0-80 percent")
        if not 3.0 <= float(integration["tm_desired_speed_mps"]) <= 12.0:
            raise ValueError("Traffic Manager desired speed must be within 3-12 m/s")
        if integration.get("reload_world_before_run") is not True:
            raise ValueError("advisor-rich corpus must reload Town10HD_Opt before every run")
        route_mode = str(integration.get("npc_route_mode", "fixed_loop"))
        if route_mode not in {"fixed_loop", "tm_autonomous", "direct_loop"}:
            raise ValueError(
                "NPC route mode must be fixed_loop, tm_autonomous, or direct_loop"
            )
        if route_mode in {"fixed_loop", "direct_loop"}:
            if str(integration["npc_loop_route_progress_csv"]) != str(
                integration["route_progress_csv"]
            ):
                raise ValueError("NPC and ego loop routes must use the same frozen progress CSV")
            if int(integration["npc_loop_repetitions"]) < 2:
                raise ValueError("NPC loop route must repeat at least twice")
        if route_mode == "direct_loop" and not 2.0 <= float(
            integration["npc_direct_route_speed_mps"]
        ) <= 10.0:
            raise ValueError("direct NPC route speed must be within 2-10 m/s")
        traffic_gate = integration.get("traffic_sanity_gate")
        if not isinstance(traffic_gate, Mapping):
            raise ValueError("advisor integration traffic_sanity_gate mapping is required")
    preflight["advisor_integration"] = {
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": base_runner._sha256(path),
            }
            for name, path in files.items()
        },
        "route_points": int(len(route["planned_path"])),
        "progress_points": int(len(progress)),
        "route_loop": True,
        "tm_port": 8010,
        "reactive_pedestrian_speed_mps": pedestrian_speed,
    }
    return preflight


def _longest_true_dwell(mask: Sequence[bool], timestamps: Sequence[float]) -> float:
    values = np.asarray(mask, dtype=bool)
    times = np.asarray(timestamps, dtype=float)
    if not len(values):
        return 0.0
    deltas = np.diff(times)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    nominal = float(np.median(deltas)) if len(deltas) else 0.0
    longest = 0.0
    start = None
    for index, enabled in enumerate(values):
        if enabled and start is None:
            start = index
        if not enabled and start is not None:
            longest = max(longest, times[index - 1] - times[start] + nominal)
            start = None
    if start is not None:
        longest = max(longest, times[-1] - times[start] + nominal)
    return float(longest)


def _controlled_pedestrian_gate_rows(
    ground_truth: pd.DataFrame, gate: Mapping[str, object]
) -> pd.DataFrame:
    """Select only the registered close-crossing realization arm."""

    in_scope = _truthy(ground_truth["in_camera_frustum"]) & (
        pd.to_numeric(ground_truth["distance_m"], errors="coerce")
        <= float(gate["headline_range_m"])
    )
    role_name = ground_truth.get(
        "role_name", pd.Series("", index=ground_truth.index)
    ).astype(str)
    pedestrian_gate_family = str(gate["pedestrian_gate_scenario_family"])
    controlled = ground_truth[
        in_scope
        & (ground_truth["scenario_family"] == pedestrian_gate_family)
        & (ground_truth["class_name"] == "pedestrian")
        & role_name.str.startswith(str(gate["pedestrian_role_prefix"]))
    ].copy()
    controlled["world_x"] = pd.to_numeric(controlled["origin_x"], errors="coerce")
    controlled["world_y"] = pd.to_numeric(controlled["origin_y"], errors="coerce")
    return controlled


def _run_smoke_gate(batch_dir: Path, config: Mapping[str, object]) -> Dict[str, object]:
    gate = config["advisor_integration"]["smoke_gate"]
    gt_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    metric_frames: List[pd.DataFrame] = []
    traffic_summaries: List[Dict[str, object]] = []
    overlay_count = 0
    for run_spec in config["smoke_runs"]:
        run_dir = batch_dir / "runs" / str(run_spec["episode_id"])
        gt = pd.read_csv(base_runner._single_csv(run_dir, "_object_ground_truth.csv"))
        pred = pd.read_csv(base_runner._single_csv(run_dir, "_object_predictions.csv"))
        metrics = pd.read_csv(base_runner._single_csv(run_dir, "_metrics.csv"))
        gt["episode_id"] = str(run_spec["episode_id"])
        gt["scenario_family"] = str(run_spec["scenario_family"])
        pred["episode_id"] = str(run_spec["episode_id"])
        pred["scenario_family"] = str(run_spec["scenario_family"])
        metrics["episode_id"] = str(run_spec["episode_id"])
        metrics["scenario_family"] = str(run_spec["scenario_family"])
        gt_frames.append(gt)
        prediction_frames.append(pred)
        metric_frames.append(metrics)
        traffic_summary_path = run_dir / "traffic_sanity" / "traffic_sanity_summary.json"
        if not traffic_summary_path.is_file():
            traffic_summaries.append(
                {
                    "episode_id": str(run_spec["episode_id"]),
                    "pass": False,
                    "failures": ["missing_traffic_sanity_summary"],
                }
            )
        else:
            traffic_summary = json.loads(traffic_summary_path.read_text(encoding="utf-8"))
            traffic_summary["episode_id"] = str(run_spec["episode_id"])
            traffic_summaries.append(traffic_summary)
        overlay_count += len(list((run_dir / "overlays").glob("*.png")))
    gt_all = pd.concat(gt_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_all = pd.concat(metric_frames, ignore_index=True)
    gt_all["class_name"] = gt_all["class_name"].map(_normalize_class)
    predictions_all["class_name"] = predictions_all["class_name"].map(_normalize_class)
    failures: List[str] = []
    classes = set(gt_all["class_name"].dropna().astype(str))
    for class_name in ("vehicle", "pedestrian"):
        if class_name not in classes:
            failures.append(f"missing_{class_name}_ground_truth")
    failed_traffic = [
        summary for summary in traffic_summaries if not bool(summary.get("pass", False))
    ]
    if failed_traffic:
        failures.append("traffic_sanity_gate_failed")

    # This is intentionally evaluated before any detection-coverage matching.
    # Requested pps/fps arguments did not catch CARLA's half-density dual-clock
    # behavior, so observed tensor support is the scale-up prerequisite.
    radar_gate = config["advisor_integration"].get("radar_density_gate")
    radar_density_by_episode: Dict[str, Dict[str, object]] = {}
    if isinstance(radar_gate, Mapping):
        for episode_id, episode_metrics in metrics_all.groupby("episode_id"):
            radar_density_by_episode[str(episode_id)] = _radar_density_summary(
                episode_metrics, radar_gate
            )
        if not radar_density_by_episode or not all(
            bool(summary["pass"]) for summary in radar_density_by_episode.values()
        ):
            failures.append("radar_density_contract_failed")

    cadence_by_episode: Dict[str, Dict[str, object]] = {}
    expected_sensor_period_s = 1.0 / float(gate["sensor_detection_hz"])
    expected_frame_step = int(gate["control_ticks_per_sensor_frame"])
    for episode_id, episode_metrics in metrics_all.groupby("episode_id"):
        episode_metrics = episode_metrics.sort_values("carla_timestamp")
        timestamp_steps = pd.to_numeric(
            episode_metrics["carla_timestamp"], errors="coerce"
        ).diff().dropna()
        frame_steps = pd.to_numeric(
            episode_metrics["frame_id"], errors="coerce"
        ).diff().dropna()
        median_period = float(timestamp_steps.median()) if len(timestamp_steps) else None
        median_frame_step = float(frame_steps.median()) if len(frame_steps) else None
        contract_fields_ok = (
            len(episode_metrics)
            and np.allclose(
                pd.to_numeric(episode_metrics["world_control_tick_hz"], errors="coerce"),
                float(gate["world_control_tick_hz"]),
            )
            and np.allclose(
                pd.to_numeric(episode_metrics["sensor_detection_hz"], errors="coerce"),
                float(gate["sensor_detection_hz"]),
            )
            and (
                pd.to_numeric(
                    episode_metrics["control_ticks_per_sensor_frame"], errors="coerce"
                )
                == expected_frame_step
            ).all()
        )
        cadence_ok = (
            median_period is not None
            and math.isclose(
                median_period,
                expected_sensor_period_s,
                rel_tol=0.0,
                abs_tol=float(gate["sensor_period_tolerance_s"]),
            )
            and median_frame_step is not None
            and math.isclose(
                median_frame_step,
                float(expected_frame_step),
                rel_tol=0.0,
                abs_tol=0.01,
            )
            and bool(contract_fields_ok)
        )
        cadence_by_episode[str(episode_id)] = {
            "pass": bool(cadence_ok),
            "median_sensor_period_s": median_period,
            "median_carla_frame_step": median_frame_step,
            "contract_fields_ok": bool(contract_fields_ok),
        }
    if not cadence_by_episode or not all(
        bool(summary["pass"]) for summary in cadence_by_episode.values()
    ):
        failures.append("sensor_control_clock_contract_failed")

    role_name = gt_all.get("role_name", pd.Series("", index=gt_all.index)).astype(str)
    pedestrian_gate_family = str(gate["pedestrian_gate_scenario_family"])
    controlled = _controlled_pedestrian_gate_rows(gt_all, gate)
    scores = pd.to_numeric(
        predictions_all.get("score", pd.Series(1.0, index=predictions_all.index)),
        errors="coerce",
    )
    predictions_all = predictions_all[scores >= float(gate["prediction_score_min"])].copy()
    matches = []
    for episode_id, episode_gt in controlled.groupby("episode_id"):
        episode_pred = predictions_all[predictions_all["episode_id"] == episode_id]
        match = _greedy_prediction_matches(
            episode_gt,
            episode_pred,
            float(gate["association_gate_m"]),
        )
        if not match.empty:
            matches.append(match)
    matched_rows = int(sum(len(frame) for frame in matches))
    pedestrian_coverage = (
        100.0 * matched_rows / len(controlled) if len(controlled) else 0.0
    )
    if len(controlled) == 0:
        failures.append("no_close_controlled_pedestrian_gt")
    if pedestrian_coverage < float(gate["minimum_controlled_pedestrian_coverage_pct"]):
        failures.append("controlled_pedestrian_coverage_below_gate")

    controlled_speed_parts = []
    for (_episode_id, _actor_id), group in controlled.groupby(
        ["episode_id", "actor_id"]
    ):
        group = group.sort_values("carla_timestamp")
        dt = pd.to_numeric(group["carla_timestamp"], errors="coerce").diff()
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        controlled_speed_parts.append(pd.Series(np.hypot(dx, dy) / dt, index=group.index))
    controlled_speed = (
        pd.concat(controlled_speed_parts).sort_index().replace([np.inf, -np.inf], np.nan).dropna()
        if controlled_speed_parts
        else pd.Series(dtype=float)
    )
    controlled_active_rows = int(
        (controlled_speed >= float(gate["minimum_controlled_pedestrian_active_speed_mps"])).sum()
    )
    if controlled_active_rows < int(gate["minimum_controlled_pedestrian_active_rows"]):
        failures.append("controlled_pedestrian_did_not_realize_crossing_motion")
    if len(controlled_speed) and controlled_speed.max() > float(gate["pedestrian_speed_max_mps"]):
        failures.append("controlled_pedestrian_speed_above_realistic_maximum")

    route_metrics = metrics_all[
        metrics_all["scenario_family"].isin(["mixed_urban", "ped_crossing"])
    ]
    route_ego_speed = pd.to_numeric(route_metrics["ego_speed_mps"], errors="coerce").dropna()
    route_ego_speed_p95 = float(route_ego_speed.quantile(0.95)) if len(route_ego_speed) else 0.0
    if route_ego_speed_p95 < float(gate["route_ego_speed_p95_min_mps"]):
        failures.append("ego_route_motion_below_gate")

    exact = gt_all[
        role_name == str(gate["exact_fast_role_name"])
    ].sort_values(["episode_id", "carla_timestamp"]).copy()
    speed_parts = []
    for _actor_id, group in exact.groupby(["episode_id", "actor_id"]):
        dt = pd.to_numeric(group["carla_timestamp"], errors="coerce").diff()
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        speed_parts.append(pd.Series(np.hypot(dx, dy) / dt, index=group.index))
    exact["derived_speed_mps"] = (
        pd.concat(speed_parts).sort_index() if speed_parts else pd.Series(dtype=float)
    )
    exact_mask = (
        _truthy(exact["in_camera_frustum"])
        & (pd.to_numeric(exact["distance_m"], errors="coerce") <= float(gate["fast_range_max_m"]))
        & (exact["derived_speed_mps"] >= float(gate["fast_speed_min_mps"]))
    )
    fast_dwell = 0.0
    for _episode_id, group in exact.groupby("episode_id"):
        group_mask = exact_mask.loc[group.index]
        fast_dwell = max(
            fast_dwell,
            _longest_true_dwell(
                group_mask.tolist(),
                pd.to_numeric(group["carla_timestamp"], errors="coerce").tolist(),
            ),
        )
    if exact.empty:
        failures.append("missing_exact_fast_target_gt")
    elif fast_dwell < float(gate["fast_dwell_min_s"]):
        failures.append("exact_fast_target_dwell_below_gate")
    exact_fast_scenario = (
        _exact_fast_scenario_summary(
            gt_all[gt_all["scenario_family"] == "exact_fast_convoy"],
            pd.read_csv(
                base_runner._resolve_repo_path(
                    str(config["advisor_integration"]["route_progress_csv"])
                )
            ),
            role_name=str(gate["exact_fast_role_name"]),
            maximum_route_offset_m=float(gate["exact_fast_max_route_offset_m"]),
            pedestrian_speed_max_mps=float(gate["pedestrian_speed_max_mps"]),
        )
        if "exact_fast_max_route_offset_m" in gate
        else {"applicable": False, "pass": True, "failures": []}
    )
    if not bool(exact_fast_scenario["pass"]):
        failures.append("exact_fast_scenario_validity_failed")
    if overlay_count < int(gate["minimum_overlay_images"]):
        failures.append("insufficient_visual_overlays")
    summary = {
        "pass": not failures,
        "failures": failures,
        "gt_classes": sorted(classes),
        "controlled_pedestrian_eligible_rows": int(len(controlled)),
        "controlled_pedestrian_scenario_family": pedestrian_gate_family,
        "controlled_pedestrian_matched_rows": matched_rows,
        "controlled_pedestrian_coverage_pct": pedestrian_coverage,
        "controlled_pedestrian_active_rows": controlled_active_rows,
        "controlled_pedestrian_speed_p50_mps": (
            float(controlled_speed.quantile(0.50)) if len(controlled_speed) else None
        ),
        "controlled_pedestrian_speed_p95_mps": (
            float(controlled_speed.quantile(0.95)) if len(controlled_speed) else None
        ),
        "controlled_pedestrian_speed_max_mps": (
            float(controlled_speed.max()) if len(controlled_speed) else None
        ),
        "route_ego_speed_p95_mps": route_ego_speed_p95,
        "legacy_pedestrian_coverage_pct": float(gate["legacy_pedestrian_coverage_pct"]),
        "exact_fast_target_rows": int(len(exact)),
        "exact_fast_dwell_s": fast_dwell,
        "exact_fast_scenario": exact_fast_scenario,
        "overlay_images": overlay_count,
        "clock_contract_by_episode": cadence_by_episode,
        "radar_density_by_episode": radar_density_by_episode,
        "traffic_sanity_by_episode": traffic_summaries,
    }
    (batch_dir / "smoke_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _validate_advisor_contract(config: Mapping[str, object]) -> None:
    base_runner._validate_collection_contract(config)
    integration = config.get("advisor_integration")
    if not isinstance(integration, Mapping):
        raise ValueError("advisor_integration mapping is required")
    pedestrian_speed = float(integration["pedestrian_speed_mps"])
    minimum_speed = float(integration["minimum_pedestrian_speed_mps"])
    if not 1.0 <= pedestrian_speed <= 2.0:
        raise ValueError("reactive pedestrian speed must remain in the pinned 1-2 m/s walking band")
    if not 0.0 <= minimum_speed <= pedestrian_speed:
        raise ValueError("minimum pedestrian speed is invalid")
    if int(integration["tm_port"]) != 8010:
        raise ValueError("advisor integration Traffic Manager port must be 8010")
    diagnostic = config.get("diagnostic_contract")
    if diagnostic is not None and not isinstance(diagnostic, Mapping):
        raise ValueError("diagnostic_contract must be a mapping")
    expected_update_hz = int(integration["update_hz"])
    if expected_update_hz not in {10, 20}:
        raise ValueError("advisor integration update_hz must be 10 or 20")
    expected_delta = 1.0 / float(expected_update_hz)
    if not math.isclose(
        float(integration["fixed_delta_seconds"]), expected_delta, abs_tol=1e-12
    ):
        raise ValueError(
            f"advisor integration fixed delta must be 1/update_hz ({expected_delta:.2f} s)"
        )
    if isinstance(diagnostic, Mapping):
        if expected_update_hz != 10:
            raise ValueError("on-contract diagnostic must run at 10 Hz")
        if config.get("runs"):
            raise ValueError("on-contract diagnostic must not define full corpus runs")
        smoke_runs = list(config.get("smoke_runs", []))
        if len(smoke_runs) != 1 or str(smoke_runs[0]["scenario_family"]) != "ped_crossing":
            raise ValueError(
                "on-contract diagnostic must contain exactly one pedestrian smoke run"
            )
    elif any(
        "--world-tick-hz" in base_runner._effective_options(
            base_runner._resolved_run_args(config, run_spec)
        )
        for run_spec in [*config.get("smoke_runs", []), *config.get("runs", [])]
    ):
        if "--safe" not in [
            str(value) for value in integration.get("common_traffic_args", [])
        ]:
            raise ValueError("advisor traffic generation must enable the --safe blueprint filter")
        if float(integration["tm_distance_to_leading_vehicle_m"]) < 2.5:
            raise ValueError("Traffic Manager following distance must be at least 2.5 m")
        if not 0.0 <= float(integration["tm_speed_difference_pct"]) <= 80.0:
            raise ValueError("Traffic Manager speed difference must be within 0-80 percent")
        if not 3.0 <= float(integration["tm_desired_speed_mps"]) <= 12.0:
            raise ValueError("Traffic Manager desired speed must be within 3-12 m/s")
        traffic_gate = integration.get("traffic_sanity_gate")
        if not isinstance(traffic_gate, Mapping):
            raise ValueError("advisor integration traffic_sanity_gate mapping is required")
        required_traffic_gate_fields = {
            "maximum_collision_incidents",
            "minimum_actor_observation_fraction",
            "stopped_speed_max_mps",
            "gridlock_minimum_npc_count",
            "gridlock_stopped_fraction",
            "persistent_gridlock_min_s",
        }
        missing_traffic_fields = sorted(required_traffic_gate_fields - set(traffic_gate))
        if missing_traffic_fields:
            raise ValueError(
                "traffic_sanity_gate is missing fields: "
                + ", ".join(missing_traffic_fields)
            )
        if expected_update_hz == 10:
            radar_gate = integration.get("radar_density_gate")
            if not isinstance(radar_gate, Mapping):
                raise ValueError("10 Hz corpus requires an observed radar_density_gate")
            required_radar_fields = {
                "reference_projected_points_median",
                "relative_tolerance",
                "minimum_metric_frames",
            }
            missing_radar_fields = sorted(required_radar_fields - set(radar_gate))
            if missing_radar_fields:
                raise ValueError(
                    "radar_density_gate is missing fields: "
                    + ", ".join(missing_radar_fields)
                )
            if float(radar_gate["reference_projected_points_median"]) <= 0.0:
                raise ValueError("radar density reference must be positive")
            if not 0.0 < float(radar_gate["relative_tolerance"]) <= 0.20:
                raise ValueError("radar density tolerance must be within (0, 0.20]")
            if int(radar_gate["minimum_metric_frames"]) <= 0:
                raise ValueError("radar density minimum frame count must be positive")
    pedestrian_locations = integration.get("pedestrian_locations", [])
    if len(pedestrian_locations) != 1 or any(
        len(location) != 4 for location in pedestrian_locations
    ):
        raise ValueError("exactly one explicit close-crossing pedestrian XYZYAW is required")
    route_path = base_runner._resolve_repo_path(str(integration["route_progress_csv"]))
    route = pd.read_csv(route_path)
    maximum_offset = float(integration["maximum_pedestrian_route_offset_m"])
    for location in pedestrian_locations:
        minimum_offset = np.hypot(
            route["ego_x"].astype(float) - float(location[0]),
            route["ego_y"].astype(float) - float(location[1]),
        ).min()
        if float(minimum_offset) > maximum_offset:
            raise ValueError(
                "pedestrian location is not close enough to the frozen UI route: "
                f"location={location}, offset={float(minimum_offset):.3f} m"
            )
    family_names = set(config.get("family_args", {}))
    if set(integration.get("families", {})) != family_names:
        raise ValueError("advisor population families must exactly match collector families")
    for run_spec in [*config.get("smoke_runs", []), *config.get("runs", [])]:
        options = base_runner._effective_options(base_runner._resolved_run_args(config, run_spec))
        for option in ("--npc-vehicles", "--npc-pedestrians"):
            if options.get(option) != "0":
                raise ValueError(f"{run_spec['episode_id']} must use observe-existing {option}=0")
        if options.get("--tm-port") != str(integration["tm_port"]):
            raise ValueError(f"{run_spec['episode_id']} collector TM port is not aligned")
        if not math.isclose(
            float(options.get("--world-tick-hz", options.get("--fps", "nan"))),
            float(integration["update_hz"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{run_spec['episode_id']} collector world/control clock is not aligned"
            )
        if expected_update_hz == 10 and not isinstance(diagnostic, Mapping):
            if options.get("--fps") != "10":
                raise ValueError(f"{run_spec['episode_id']} sensor clock must be 10 Hz")
            if options.get("--sensor-every-tick") != "true":
                raise ValueError(
                    f"{run_spec['episode_id']} must emit sensors on every 10 Hz world tick"
                )
            if "--no-sensor-every-tick" in options:
                raise ValueError(
                    f"{run_spec['episode_id']} cannot use the dual-clock sensor skip mode"
                )
        if options.get("--ego-fixed-path-progress-csv") != str(integration["route_progress_csv"]):
            raise ValueError(f"{run_spec['episode_id']} does not use the frozen advisor route")
        if options.get("--ego-spawn-index") != str(integration["ego_spawn_index"]):
            raise ValueError(f"{run_spec['episode_id']} collector spawn differs from reservation")
        walker_ignore = float(options.get("--ego-ignore-walkers-pct", "0"))
        route_control = str(options.get("--ego-route-control", "traffic_manager"))
        family = str(run_spec["scenario_family"])
        expected_route_control = (
            "traffic_manager" if family == "exact_fast_convoy" else "direct"
        )
        if route_control != expected_route_control:
            raise ValueError(
                f"{run_spec['episode_id']} must use {expected_route_control} ego route control"
            )
        if family == "ped_crossing":
            if not math.isclose(walker_ignore, 100.0, abs_tol=1e-12):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must use the pinned "
                    "100% ego walker-ignore exception"
                )
            if options.get("--ego-direct-yield-to-controlled-pedestrian") != "true":
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must use direct ego yield"
                )
            if not math.isclose(
                float(options.get("--ego-spawn-forward-offset-m", "0")),
                14.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must start at the pinned "
                    "close-route offset"
                )
            if not math.isclose(
                float(options.get("--ego-direct-route-speed-mps", "0")),
                3.5,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must use the pinned "
                    "3.5 m/s urban ego speed"
                )
            if "--braking-margin" not in integration["families"][family].get(
                "blocker_args", []
            ):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian blocker must use the pinned "
                    "conservative arming margin"
                )
        elif not math.isclose(walker_ignore, 0.0, abs_tol=1e-12):
            raise ValueError(
                f"{run_spec['episode_id']} walker-ignore exception leaked outside ped_crossing"
            )
        elif options.get("--ego-direct-yield-to-controlled-pedestrian") == "true":
            raise ValueError(
                f"{run_spec['episode_id']} pedestrian-yield control leaked outside ped_crossing"
            )
    reservation_indices = [int(value) for value in integration["ego_reservation_spawn_indices"]]
    if int(integration["ego_spawn_index"]) not in reservation_indices:
        raise ValueError("ego reservation corridor must include the collector spawn")
    if len(reservation_indices) != len(set(reservation_indices)):
        raise ValueError("ego reservation spawn indices must be unique")


def _load_config(path: Path) -> Dict[str, object]:
    config = base_runner._load_config(path)
    _validate_advisor_contract(config)
    return config


def run_batch(
    config_path: Path,
    mode: str,
    batch_dir: Path | None,
    dry_run: bool,
    only_episode_ids: Sequence[str] = (),
) -> Path:
    config = _load_config(config_path)
    selected_runs: Iterable[Mapping[str, object]] = (
        config["smoke_runs"] if mode == "smoke" else config["runs"]
    )
    selected_runs = list(selected_runs)
    if only_episode_ids:
        wanted = set(only_episode_ids)
        selected_runs = [
            item for item in selected_runs if str(item["episode_id"]) in wanted
        ]
        found = {str(item["episode_id"]) for item in selected_runs}
        if found != wanted:
            raise ValueError("unknown --only-episode values: " + ", ".join(sorted(wanted - found)))
    preflight = _static_preflight(config)
    client = world = None
    if not dry_run:
        client, world = _connect(config)
        preflight["live_carla"] = _require_empty_async(world)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if batch_dir is None:
        batch_dir = base_runner._resolve_repo_path(str(config["output_root"])) / f"{timestamp}_{mode}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    resolved_config_path = batch_dir / "resolved_collection_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest: MutableMapping[str, object] = {
        "schema": "policy_corpus_batch.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "dry_run" if dry_run else "running",
        "config_path": str(config_path),
        "config_sha256": base_runner._sha256(config_path),
        "batch_dir": str(batch_dir),
        "preflight": preflight,
        "runs": [],
    }
    manifest_path = batch_dir / "batch_manifest.json"
    base_runner._write_manifest(manifest_path, manifest)

    integration = config["advisor_integration"]
    for run_spec in selected_runs:
        episode_id = str(run_spec["episode_id"])
        family = str(run_spec["scenario_family"])
        family_spec = integration["families"][family]
        run_dir = batch_dir / "runs" / episode_id
        collector_command = base_runner._run_command(config, run_spec, run_dir)
        blocker_command, traffic_command = _population_commands(config, run_spec)
        record: MutableMapping[str, object] = {
            **dict(run_spec),
            "command": collector_command,
            "blocker_command": blocker_command,
            "traffic_command": traffic_command,
            "run_dir": str(run_dir),
            "status": "planned" if dry_run else "running",
        }
        manifest["runs"].append(record)
        base_runner._write_manifest(manifest_path, manifest)
        if dry_run:
            continue
        assert client is not None and world is not None
        print(
            f"[{episode_id}] configuring single "
            f"{int(integration['update_hz'])} Hz sync master",
            flush=True,
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        processes: List[Tuple[str, subprocess.Popen, object]] = []
        ego_reservations: List[carla.Actor] = []
        traffic_monitor: TrafficSanityMonitor | None = None
        original_settings = None
        collector_result = None
        try:
            if integration.get("reload_world_before_run") is True:
                world = client.load_world(str(config["carla"]["expected_town"]), True)
                record["world_reload"] = {
                    "requested_map": str(config["carla"]["expected_town"]),
                    "resolved_map": str(world.get_map().name),
                }
            _require_empty_async(world)
            original_settings = _set_sync_master(
                client,
                world,
                int(integration["tm_port"]),
                float(integration["fixed_delta_seconds"]),
            )
            ego_reservations = _spawn_ego_reservations(
                world,
                [int(value) for value in integration["ego_reservation_spawn_indices"]],
            )
            record["ego_spawn_reservation"] = {
                "actor_ids": [int(actor.id) for actor in ego_reservations],
                "spawn_indices": [
                    int(value) for value in integration["ego_reservation_spawn_indices"]
                ],
            }
            populator_processes: List[subprocess.Popen] = []
            if blocker_command:
                blocker, blocker_stream = _start_process(
                    blocker_command, run_dir / "spawn_blocker.log"
                )
                processes.append(("spawn_blocker_v4", blocker, blocker_stream))
                populator_processes.append(blocker)
                record["blocker_ready"] = _tick_until(
                    world,
                    [blocker],
                    lambda inventory, roles: _blocker_ready(
                        inventory, roles, family_spec
                    ),
                    float(integration["population_start_timeout_s"]),
                    "spawn_blocker_v4 readiness",
                )
            else:
                record["blocker_ready"] = {
                    "applicable": False,
                    "reason": "all blocker categories disabled for scenario family",
                }
            traffic, traffic_stream = _start_process(
                traffic_command, run_dir / "generate_traffic.log"
            )
            processes.append(("generate_traffic_v1", traffic, traffic_stream))
            populator_processes.append(traffic)
            record["population_ready"] = _tick_until(
                world,
                populator_processes,
                lambda inventory, roles: _population_ready(inventory, roles, family_spec),
                float(integration["population_start_timeout_s"]),
                "advisor traffic population",
            )
            _destroy_ego_reservations(world, ego_reservations)
            ego_reservations = []
            record["ego_spawn_reservation"]["released_before_collector"] = True
            if not isinstance(config.get("diagnostic_contract"), Mapping):
                traffic_monitor = TrafficSanityMonitor(
                    world=world,
                    traffic_manager=client.get_trafficmanager(int(integration["tm_port"])),
                    output_dir=run_dir / "traffic_sanity",
                    integration=integration,
                )
                traffic_monitor.start()
                record["traffic_sanity_initial_geometry"] = traffic_monitor.initial_geometry
            print(
                f"[{episode_id}] population ready; yielding sole tick ownership to collector",
                flush=True,
            )
            log_path = run_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log_stream:
                collector_result = subprocess.run(
                    collector_command,
                    cwd=REPO_ROOT,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=float(integration["collector_timeout_s"]),
                )
            record["returncode"] = int(collector_result.returncode)
        except Exception as exc:
            record["orchestration_error"] = f"{type(exc).__name__}: {exc}"
            record["status"] = "orchestration_failed"
            manifest["status"] = "failed"
            raise
        finally:
            if original_settings is not None:
                restored_async = False
                try:
                    _destroy_ego_reservations(world, ego_reservations)
                    ego_reservations = []
                    if traffic_monitor is not None:
                        record["traffic_sanity"] = traffic_monitor.stop()
                        traffic_monitor = None
                    client.get_trafficmanager(int(integration["tm_port"])).set_synchronous_mode(True)
                    record["populator_shutdown"] = _stop_processes(
                        world,
                        processes,
                        float(integration["population_shutdown_timeout_s"]),
                    )
                    # CARLA 0.10 can defer destruction of a returned collector's
                    # attached sensors/ego until the next asynchronous frame.
                    # Restore the clock before passive postflight polling; the
                    # previous sync-first ordering misclassified this bounded
                    # teardown latency as a leak while the world was frozen.
                    _restore_async(
                        client,
                        world,
                        int(integration["tm_port"]),
                        original_settings,
                    )
                    restored_async = True
                    record["postflight_dynamic_actor_counts"] = _tick_until_empty(
                        world, float(integration["population_shutdown_timeout_s"])
                    )
                finally:
                    if not restored_async:
                        _restore_async(
                            client,
                            world,
                            int(integration["tm_port"]),
                            original_settings,
                        )
                    record["restored_world"] = _require_empty_async(world)
            base_runner._write_manifest(manifest_path, manifest)

        assert collector_result is not None
        log_path = run_dir / "run.log"
        try:
            record["basic_gate"] = base_runner._basic_run_gate(
                run_dir, run_spec, config
            )
            metrics = pd.read_csv(base_runner._single_csv(run_dir, "_metrics.csv"))
            radar_gate = config["advisor_integration"].get("radar_density_gate")
            record["radar_density_gate"] = (
                _radar_density_summary(metrics, radar_gate)
                if isinstance(radar_gate, Mapping)
                else {"applicable": False, "pass": True, "failures": []}
            )
            smoke_gate = config["advisor_integration"]["smoke_gate"]
            ground_truth = pd.read_csv(
                base_runner._single_csv(run_dir, "_object_ground_truth.csv")
            )
            record["pedestrian_motion_gate"] = _pedestrian_motion_summary(
                ground_truth,
                maximum_speed_mps=float(smoke_gate["pedestrian_speed_max_mps"]),
            )
            if (
                family == "exact_fast_convoy"
                and "exact_fast_max_route_offset_m" in smoke_gate
            ):
                record["exact_fast_scenario_gate"] = _exact_fast_scenario_summary(
                    ground_truth,
                    pd.read_csv(
                        base_runner._resolve_repo_path(
                            str(integration["route_progress_csv"])
                        )
                    ),
                    role_name=str(
                        smoke_gate["exact_fast_role_name"]
                    ),
                    maximum_route_offset_m=float(
                        smoke_gate["exact_fast_max_route_offset_m"]
                    ),
                    pedestrian_speed_max_mps=float(
                        smoke_gate["pedestrian_speed_max_mps"]
                    ),
                )
            else:
                record["exact_fast_scenario_gate"] = {
                    "applicable": False,
                    "pass": True,
                    "failures": [],
                }
        except Exception as exc:
            record["status"] = "basic_gate_error"
            record["basic_gate_error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"
            base_runner._write_manifest(manifest_path, manifest)
            raise
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4096:]
        known_teardown_abort = (
            collector_result.returncode == -6
            and "libc++abi" in log_tail
            and "std::exception" in log_tail
        )
        record["known_carla_teardown_abort"] = known_teardown_abort
        accepted = collector_result.returncode == 0 or (
            known_teardown_abort and record["basic_gate"]["pass"]
        )
        # The smoke authorizes scale-up; the same traffic contract still applies
        # to every collected trajectory. Fail immediately so a collision cannot
        # enter a corpus and be discovered only after all remaining runs finish.
        traffic_sane = bool(record.get("traffic_sanity", {}).get("pass", True))
        radar_dense = bool(record["radar_density_gate"]["pass"])
        exact_fast_valid = bool(record["exact_fast_scenario_gate"]["pass"])
        pedestrian_motion_valid = bool(record["pedestrian_motion_gate"]["pass"])
        if (
            record["basic_gate"]["pass"]
            and accepted
            and traffic_sane
            and radar_dense
            and exact_fast_valid
            and pedestrian_motion_valid
        ):
            record["status"] = (
                "complete" if collector_result.returncode == 0 else "complete_with_teardown_warning"
            )
        else:
            if (
                not record["basic_gate"]["pass"]
                or not traffic_sane
                or not radar_dense
                or not exact_fast_valid
                or not pedestrian_motion_valid
            ):
                record["status"] = "gate_failed"
            else:
                record["status"] = "collector_failed"
            manifest["status"] = "failed"
            base_runner._write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"{episode_id} failed: returncode={collector_result.returncode}, "
                f"basic_gate={record['basic_gate']}, "
                f"radar_density_gate={record['radar_density_gate']}, "
                f"exact_fast_scenario_gate={record['exact_fast_scenario_gate']}, "
                f"pedestrian_motion_gate={record['pedestrian_motion_gate']}, "
                f"traffic_sanity={record.get('traffic_sanity')}"
            )
        base_runner._write_manifest(manifest_path, manifest)
        print(f"[{episode_id}] complete and actor-clean", flush=True)

    if dry_run:
        base_runner._write_manifest(manifest_path, manifest)
        return batch_dir
    if mode == "smoke" and not only_episode_ids:
        smoke_gate = _run_smoke_gate(batch_dir, config)
        manifest["smoke_gate"] = smoke_gate
        manifest["status"] = "smoke_pass" if smoke_gate["pass"] else "smoke_gate_failed"
        base_runner._write_manifest(manifest_path, manifest)
        if not smoke_gate["pass"]:
            raise RuntimeError("advisor-rich smoke gate failed: " + ", ".join(smoke_gate["failures"]))
    elif isinstance(config.get("diagnostic_contract"), Mapping):
        manifest["status"] = "diagnostic_capture_complete_pending_replay"
        base_runner._write_manifest(manifest_path, manifest)
    else:
        manifest["status"] = "collection_complete_pending_verification"
        base_runner._write_manifest(manifest_path, manifest)
    return batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--only-episode", action="append", default=[])
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.validate_config:
        config = _load_config(config_path)
        print(
            json.dumps(
                {
                    "experiment_name": config["experiment_name"],
                    "full_runs": len(config.get("runs", [])),
                    "smoke_runs": len(config.get("smoke_runs", [])),
                    "status": "VALID",
                },
                sort_keys=True,
            )
        )
        return
    print(
        run_batch(
            config_path,
            args.mode,
            args.batch_dir,
            args.dry_run,
            args.only_episode,
        )
    )


if __name__ == "__main__":
    # This shipping libcarla aborts its first RPC when a CARLA-owning module is
    # loaded directly by runpy. Replace the runpy process rather than keeping
    # that libcarla-loaded parent alive; on L10319 the parent/child form can
    # make every RPC in the child fail with ``Operation aborted``.
    os.chdir(REPO_ROOT)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-c",
            "from data_collection.run_advisor_policy_corpus import main; main()",
            *sys.argv[1:],
        ],
    )

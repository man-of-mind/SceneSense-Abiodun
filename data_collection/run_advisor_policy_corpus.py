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
import bisect
import csv
import hashlib
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
    client = carla.Client(str(connection["host"]), int(connection["port"]))
    client.set_timeout(float(connection.get("timeout_s", 10.0)))
    for _attempt in range(attempts):
        try:
            # This CARLA 0.10 Linux package intermittently aborts get_world()
            # when it is the first RPC on a fresh client. The lightweight
            # version request normally establishes the session. Retaining the
            # same client across bounded retries avoids a loop of perpetually
            # fresh sessions when the server returns ``Operation aborted``.
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
    expected_frame_count: Optional[int] = None,
    stationary_context_expected: bool = False,
    stationary_context_trajectories: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Summarize NPC collisions and sustained network-wide gridlock."""

    expected_ids = {int(value) for value in expected_actor_ids}
    trajectories = trajectories.copy()
    collisions = collisions.copy()
    observed_ids = set(
        pd.to_numeric(trajectories.get("actor_id", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
    )
    observation_fraction = (
        len(expected_ids & observed_ids) / len(expected_ids)
        if expected_ids
        else 1.0
    )
    observed_frames_by_actor = {
        actor_id: int(
            trajectories.loc[
                pd.to_numeric(trajectories.get("actor_id"), errors="coerce").eq(actor_id),
                "frame_id",
            ].nunique()
        )
        for actor_id in sorted(expected_ids)
    }
    per_actor_frame_fraction = None
    if expected_frame_count is not None and expected_ids:
        if int(expected_frame_count) <= 0:
            raise ValueError("traffic expected frame count must be positive")
        per_actor_frame_fraction = min(observed_frames_by_actor.values(), default=0) / int(
            expected_frame_count
        )

    collision_incidents = 0
    ignored_static_contact_rows = 0
    collision_events_by_owner_scope: Dict[str, int] = {}
    if not collisions.empty:
        collision_rows = collisions.copy()
        collision_rows["frame_id"] = pd.to_numeric(
            collision_rows["frame_id"], errors="coerce"
        ).fillna(-1).astype(int)
        first = pd.to_numeric(collision_rows["npc_actor_id"], errors="coerce").fillna(-1).astype(int)
        second = pd.to_numeric(collision_rows["other_actor_id"], errors="coerce").fillna(-1).astype(int)
        other_type = collision_rows.get(
            "other_type_id", pd.Series("", index=collision_rows.index, dtype=str)
        ).fillna("").astype(str)
        collision_rows["other_type_id"] = other_type
        impulse_x = pd.to_numeric(
            collision_rows.get(
                "normal_impulse_x", pd.Series(0.0, index=collision_rows.index)
            ),
            errors="coerce",
        ).fillna(0.0)
        impulse_y = pd.to_numeric(
            collision_rows.get(
                "normal_impulse_y", pd.Series(0.0, index=collision_rows.index)
            ),
            errors="coerce",
        ).fillna(0.0)
        horizontal_impulse = np.hypot(impulse_x, impulse_y)
        static_horizontal_gate = float(
            gate.get("minimum_static_collision_horizontal_impulse", 50.0)
        )
        # CARLA collision sensors report benign body/ground and walker/sidewalk
        # contacts, including high *vertical* suspension impulses.  Count every
        # actor-to-actor contact, plus only static-world contacts with material
        # horizontal impulse.  This preserves vehicle/pedestrian and vehicle/
        # obstacle failures without treating gravity settlement as a crash.
        relevant = (second > 0) | (
            other_type.str.startswith("static.")
            & (horizontal_impulse >= static_horizontal_gate)
        )
        ignored_static_contact_rows = int((~relevant).sum())
        collision_rows = collision_rows.loc[relevant].copy()
        first = first.loc[relevant]
        second = second.loc[relevant]
        owner_scope = collision_rows.get(
            "contact_owner_scope",
            pd.Series("ambient_npc", index=collision_rows.index, dtype=str),
        ).fillna("unknown").astype(str)
        collision_rows["pair_low"] = np.minimum(first, second)
        collision_rows["pair_high"] = np.maximum(first, second)
        collision_rows["owner_scope"] = owner_scope
        collision_rows = collision_rows.sort_values(
            ["owner_scope", "pair_low", "pair_high", "other_type_id", "frame_id"]
        )
        group_fields = ["owner_scope", "pair_low", "pair_high", "other_type_id"]
        previous_frame = collision_rows.groupby(group_fields)["frame_id"].shift()
        new_incident = previous_frame.isna() | (
            (collision_rows["frame_id"] - previous_frame) > 10
        )
        collision_incidents = int(new_incident.sum())
        collision_events_by_owner_scope = {
            str(scope): int(count)
            for scope, count in collision_rows.loc[new_incident]
            .groupby("owner_scope")
            .size()
            .items()
        }

    persistent_gridlock_dwell_s = 0.0
    raw_stopped_network_dwell_s = 0.0
    registered_hazard_yield_dwell_s = 0.0
    npc_speed_p50_mps = None
    stopped_fraction_p95 = None
    unattributed_stopped_fraction_p95 = None
    path_distance_by_actor_m: Dict[str, float] = {}
    if not trajectories.empty:
        trajectories["speed_mps"] = pd.to_numeric(
            trajectories["speed_mps"], errors="coerce"
        )
        trajectories["registered_hazard_yield_active"] = trajectories.get(
            "registered_hazard_yield_active",
            pd.Series(False, index=trajectories.index),
        ).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
        valid_speed = trajectories["speed_mps"].dropna()
        if len(valid_speed):
            npc_speed_p50_mps = float(valid_speed.quantile(0.50))
        if {"world_x", "world_y"}.issubset(trajectories.columns):
            for actor_id, actor_rows in trajectories.groupby("actor_id"):
                ordered = actor_rows.sort_values("frame_id")
                step_distance = np.hypot(
                    pd.to_numeric(ordered["world_x"], errors="coerce").diff(),
                    pd.to_numeric(ordered["world_y"], errors="coerce").diff(),
                )
                path_distance_by_actor_m[str(int(actor_id))] = float(
                    step_distance.fillna(0.0).sum()
                )
        trajectories["unattributed_stopped"] = (
            trajectories["speed_mps"] <= float(gate["stopped_speed_max_mps"])
        ) & ~trajectories["registered_hazard_yield_active"]
        per_frame = trajectories.groupby("frame_id").agg(
            carla_timestamp=("carla_timestamp", "median"),
            npc_count=("actor_id", "nunique"),
            stopped_fraction=(
                "speed_mps",
                lambda values: float(
                    (values <= float(gate["stopped_speed_max_mps"])).mean()
                ),
            ),
            registered_hazard_yield_active=(
                "registered_hazard_yield_active", "max"
            ),
            unattributed_stopped_fraction=(
                "unattributed_stopped", "mean"
            ),
        ).sort_index()
        if len(per_frame):
            stopped_fraction_p95 = float(per_frame["stopped_fraction"].quantile(0.95))
            unattributed_stopped_fraction_p95 = float(
                per_frame["unattributed_stopped_fraction"].quantile(0.95)
            )
            raw_gridlocked = (
                per_frame["npc_count"] >= int(gate["gridlock_minimum_npc_count"])
            ) & (
                per_frame["stopped_fraction"]
                >= float(gate["gridlock_stopped_fraction"])
            )
            raw_stopped_network_dwell_s = _longest_mask_dwell_s(
                raw_gridlocked, per_frame["carla_timestamp"]
            )
            registered_hazard_yield_dwell_s = _longest_mask_dwell_s(
                per_frame["registered_hazard_yield_active"],
                per_frame["carla_timestamp"],
            )
            # Exempt only the individual actors geometrically yielding to the
            # registered hazard. A single legitimate yield must not mask a
            # different, network-wide stall (the old frame-global Boolean did).
            gridlocked = (
                per_frame["npc_count"] >= int(gate["gridlock_minimum_npc_count"])
            ) & (
                per_frame["unattributed_stopped_fraction"]
                >= float(gate["gridlock_stopped_fraction"])
            )
            persistent_gridlock_dwell_s = _longest_mask_dwell_s(
                gridlocked, per_frame["carla_timestamp"]
            )

    stationary_context_path_distance_by_actor_m = dict(path_distance_by_actor_m)
    if stationary_context_trajectories is not None:
        stationary_rows = stationary_context_trajectories.copy()
        stationary_context_path_distance_by_actor_m = {}
        if (
            not stationary_rows.empty
            and {"actor_id", "frame_id", "world_x", "world_y"}.issubset(
                stationary_rows.columns
            )
        ):
            for actor_id, actor_rows in stationary_rows.groupby("actor_id"):
                ordered = actor_rows.sort_values("frame_id")
                step_distance = np.hypot(
                    pd.to_numeric(ordered["world_x"], errors="coerce").diff(),
                    pd.to_numeric(ordered["world_y"], errors="coerce").diff(),
                )
                stationary_context_path_distance_by_actor_m[
                    str(int(actor_id))
                ] = float(step_distance.fillna(0.0).sum())

    failures: List[str] = []
    if observation_fraction < float(gate["minimum_actor_observation_fraction"]):
        failures.append("insufficient_npc_trajectory_observation")
    if (
        per_actor_frame_fraction is not None
        and per_actor_frame_fraction
        < float(gate.get("minimum_per_actor_frame_observation_fraction", 0.0))
    ):
        failures.append("insufficient_npc_per_frame_observation")
    if collision_incidents > int(gate["maximum_collision_incidents"]):
        failures.append("owned_actor_collision_incidents_above_gate")
    maximum_stationary_path_distance_m = (
        max(stationary_context_path_distance_by_actor_m.values(), default=0.0)
        if stationary_context_expected
        else None
    )
    if (
        stationary_context_expected
        and maximum_stationary_path_distance_m
        > float(gate["maximum_stationary_context_path_distance_m"])
    ):
        failures.append("stationary_context_moved")
    if (
        not stationary_context_expected
        and persistent_gridlock_dwell_s >= float(gate["persistent_gridlock_min_s"])
    ):
        failures.append("persistent_network_gridlock")
    return {
        "applicable": bool(
            expected_ids or stationary_context_expected or not collisions.empty
        ),
        "pass": not failures,
        "failures": failures,
        "monitored_npc_vehicles": int(len(expected_ids)),
        "observed_npc_vehicles": int(len(expected_ids & observed_ids)),
        "actor_observation_fraction": float(observation_fraction),
        "expected_frame_count": expected_frame_count,
        "observed_frames_by_actor": {
            str(actor_id): frames
            for actor_id, frames in observed_frames_by_actor.items()
        },
        "minimum_per_actor_frame_observation_fraction": per_actor_frame_fraction,
        "collision_callback_rows": int(len(collisions)),
        "ignored_static_contact_rows": ignored_static_contact_rows,
        "collision_events": int(collision_incidents),
        "collision_events_by_owner_scope": collision_events_by_owner_scope,
        "persistent_gridlock_dwell_s": float(persistent_gridlock_dwell_s),
        "gridlock_gate_applicable": not stationary_context_expected,
        "stationary_context_expected": bool(stationary_context_expected),
        "maximum_stationary_context_path_distance_m": (
            maximum_stationary_path_distance_m
        ),
        "raw_stopped_network_dwell_s": float(raw_stopped_network_dwell_s),
        "registered_hazard_yield_dwell_s": float(
            registered_hazard_yield_dwell_s
        ),
        "npc_speed_p50_mps": npc_speed_p50_mps,
        "stopped_fraction_p95": stopped_fraction_p95,
        "unattributed_stopped_fraction_p95": unattributed_stopped_fraction_p95,
        "path_distance_by_actor_m": path_distance_by_actor_m,
        "stationary_context_path_distance_by_actor_m": (
            stationary_context_path_distance_by_actor_m
        ),
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
    *,
    registered_crossing: bool = False,
) -> bool:
    speed_mps = math.hypot(float(walker_velocity.x), float(walker_velocity.y))
    # A controlled registered target has an authored crossing intent before it
    # starts moving. Reserve that crossing corridor while it waits at the curb;
    # otherwise an NPC queue can stop across the path and the walker will enter
    # the side of an already stationary vehicle. Incidental stationary ambient
    # pedestrians retain the narrow limit so dense sidewalks do not gridlock.
    lateral_limit = (
        5.5
        if registered_crossing
        else (3.5 if speed_mps >= 0.2 else 2.2)
    )
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
    """Yield to the occupied travel lane without blocking on an adjacent lane.

    A prior widening corridor reached 6 m at 12 m look-ahead and therefore
    treated a controlled occluder one full 3.5 m lane away as a lead vehicle.
    The reviewed direct routes have sufficiently dense waypoints that a 2.6 m
    centre corridor covers their curvature while remaining below lane-centre
    separation.
    """

    location = transform.location
    forward = transform.get_forward_vector()
    dx = float(vehicle_location.x) - float(location.x)
    dy = float(vehicle_location.y) - float(location.y)
    forward_m = dx * float(forward.x) + dy * float(forward.y)
    lateral_m = abs(-dx * float(forward.y) + dy * float(forward.x))
    return bool(
        0.0 < forward_m <= float(maximum_forward_m)
        and lateral_m <= 2.6
    )


def _vehicle_envelope_requires_yield(
    actor: object,
    transform: object,
    other: object,
    *,
    speed_mps: float,
    maximum_forward_m: float = 12.0,
) -> bool:
    """Envelope-aware vehicle shield for crossing and adjacent-lane traffic.

    Same-heading traffic uses its narrow physical width, so an adjacent lane
    does not become a false lead.  A crossing/turning vehicle contributes its
    rotated length plus a 4 m prediction margin (0.8 s at the bounded 5 m/s
    approach), allowing the shield to brake before the two envelopes overlap.
    """

    other_transform = other.get_transform()
    location = transform.location
    other_location = other_transform.location
    forward = transform.get_forward_vector()
    dx = float(other_location.x) - float(location.x)
    dy = float(other_location.y) - float(location.y)
    forward_m = dx * float(forward.x) + dy * float(forward.y)
    lateral_m = abs(-dx * float(forward.y) + dy * float(forward.x))
    own_box = getattr(actor, "bounding_box", None)
    other_box = getattr(other, "bounding_box", None)
    own_half_length_m = float(own_box.extent.x) if own_box is not None else 2.5
    own_half_width_m = float(own_box.extent.y) if own_box is not None else 1.0
    other_half_length_m = (
        float(other_box.extent.x) if other_box is not None else 2.5
    )
    other_half_width_m = (
        float(other_box.extent.y) if other_box is not None else 1.0
    )
    relative_yaw = (
        math.radians(
            float(other_transform.rotation.yaw) - float(transform.rotation.yaw)
        )
        + math.pi
    ) % (2.0 * math.pi) - math.pi
    effective_other_half_length_m = (
        abs(math.cos(relative_yaw)) * other_half_length_m
        + abs(math.sin(relative_yaw)) * other_half_width_m
    )
    effective_other_half_width_m = (
        abs(math.sin(relative_yaw)) * other_half_length_m
        + abs(math.cos(relative_yaw)) * other_half_width_m
    )
    crossing_prediction_margin_m = (
        4.0 if abs(relative_yaw) >= math.radians(15.0) else 0.0
    )
    lateral_limit_m = (
        own_half_width_m
        + effective_other_half_width_m
        + 0.4
        + crossing_prediction_margin_m
    )
    stopping_m = min(
        float(maximum_forward_m),
        max(
            7.0,
            own_half_length_m
            + effective_other_half_length_m
            + 2.0
            + float(speed_mps) ** 2 / 5.0,
        ),
    )
    return bool(0.0 < forward_m <= stopping_m and lateral_m <= lateral_limit_m)


class TrafficSanityMonitor:
    """Trajectory logger, bounded NPC controller, and collision monitor.

    A synchronous orchestrator may set ``external_sync_tick_owner`` and call
    :meth:`before_world_tick` / :meth:`observe_snapshot` itself.  This avoids
    relying on CARLA's asynchronous ``on_tick`` callback thread while another
    client owns the synchronous clock.
    """

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
        self.ambient_actor_trajectory_rows: List[Dict[str, object]] = []
        self.collision_rows: List[Dict[str, object]] = []
        self._tick_callback_id = None
        self._lock = threading.Lock()
        self.initial_geometry: Dict[str, object] = {}
        self._direct_route_state: Dict[int, Dict[str, object]] = {}
        self._direct_route_speed_mps: float | None = None
        self._replay_route_state: Dict[int, Dict[str, object]] = {}
        self._stationary_context_state: Dict[int, Dict[str, object]] = {}
        self._replay_speed_mps: float | None = None
        self._replay_fixed_delta_seconds: float | None = None
        self._route_mode = "fixed_loop"
        self._tick_failure: str | None = None
        self._registered_hazard_yield_actor_ids: set[int] = set()
        self._ambient_walker_ids: List[int] = []
        self._ambient_walker_metadata: Dict[int, Dict[str, object]] = {}
        self._observed_tick_count = 0

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

    def _live_vehicle_map(self) -> Dict[int, carla.Actor]:
        """Return authoritative membership from the server-backed inventory.

        ``world.get_actor(id)`` can yield a cached tombstone after a separately
        owned population process destroys an actor. The full actor inventory is
        the liveness boundary for this monitor.
        """

        return {
            int(actor.id): actor
            for actor in self.world.get_actors().filter("vehicle.*")
        }

    def _npc_loop_routes(self) -> List[List[carla.Location]]:
        raw_paths = self.integration.get("npc_loop_route_progress_csvs")
        if raw_paths is None:
            raw_paths = [self.integration["npc_loop_route_progress_csv"]]
        if isinstance(raw_paths, (str, Path)):
            raw_paths = [raw_paths]
        routes: List[List[carla.Location]] = []
        for raw_path in raw_paths:
            path = base_runner._resolve_repo_path(str(raw_path))
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
            routes.append(locations)
        if not routes:
            raise RuntimeError("at least one NPC loop route is required")
        return routes

    @staticmethod
    def _nearest_loop_route(
        actor_location: carla.Location,
        routes: Sequence[Sequence[carla.Location]],
    ) -> Sequence[carla.Location]:
        if not routes:
            raise ValueError("cannot assign an NPC without a route")
        return min(
            routes,
            key=lambda route: min(actor_location.distance(point) for point in route),
        )

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
    def _route_is_closed(
        route: Sequence[carla.Location], maximum_endpoint_gap_m: float = 10.0
    ) -> bool:
        return bool(
            len(route) >= 2
            and route[0].distance(route[-1]) <= float(maximum_endpoint_gap_m)
        )

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

    @staticmethod
    def _deterministic_waypoint_choice(
        candidates: Sequence[object], previous_yaw_deg: float
    ) -> object:
        """Choose one legal continuation without depending on CARLA list order."""

        if not candidates:
            raise RuntimeError("native-lane replay route reached a dead end")

        def key(waypoint: object) -> tuple[float, int, int, float, float]:
            transform = waypoint.transform
            yaw_error = abs(
                (
                    float(transform.rotation.yaw)
                    - float(previous_yaw_deg)
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            return (
                yaw_error,
                int(getattr(waypoint, "road_id", 0)),
                int(getattr(waypoint, "lane_id", 0)),
                round(float(transform.location.x), 6),
                round(float(transform.location.y), 6),
            )

        return min(candidates, key=key)

    def _build_native_lane_replay(
        self, actor: carla.Actor, *, horizon_m: float, step_m: float
    ) -> Dict[str, object]:
        """Build an immutable native-lane trace from a held actor pose.

        Ambient traffic in the paired audit must have the same future in both
        positive and benign arms.  Traffic Manager cannot provide that
        guarantee because its junction reservations depend on timing and on
        the arm-specific controlled actors.  This route is therefore resolved
        once, before capture, and replayed kinematically by the sole sync owner.
        """

        actor_transform = actor.get_transform()
        road_map = self.world.get_map()
        waypoint = road_map.get_waypoint(
            actor_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError(
                f"ambient NPC {int(actor.id)} is not on a native driving lane"
            )
        # Do not preserve suspension-settle Z: CARLA can settle the same held
        # vehicle at slightly different vertical origins in two world reloads.
        # Native waypoint Z is the deterministic road-reference contract.
        height_offset = 0.0
        canonical_start = carla.Transform(
            carla.Location(
                x=float(actor_transform.location.x),
                y=float(actor_transform.location.y),
                z=float(waypoint.transform.location.z),
            ),
            carla.Rotation(
                pitch=float(waypoint.transform.rotation.pitch),
                yaw=float(actor_transform.rotation.yaw),
                roll=float(waypoint.transform.rotation.roll),
            ),
        )
        transforms = [canonical_start]
        cumulative = [0.0]
        previous_yaw = float(actor_transform.rotation.yaw)
        current = waypoint
        while cumulative[-1] < float(horizon_m):
            current = self._deterministic_waypoint_choice(
                list(current.next(float(step_m))), previous_yaw
            )
            source = current.transform
            transform = carla.Transform(
                carla.Location(
                    x=float(source.location.x),
                    y=float(source.location.y),
                    z=float(source.location.z) + height_offset,
                ),
                carla.Rotation(
                    pitch=float(source.rotation.pitch),
                    yaw=float(source.rotation.yaw),
                    roll=float(source.rotation.roll),
                ),
            )
            segment_m = transforms[-1].location.distance(transform.location)
            if segment_m <= 1e-4:
                raise RuntimeError("native-lane replay produced a zero-length segment")
            transforms.append(transform)
            cumulative.append(cumulative[-1] + float(segment_m))
            previous_yaw = float(transform.rotation.yaw)
        serialized = [
            [
                round(float(item.location.x), 5),
                round(float(item.location.y), 5),
                round(float(item.location.z), 5),
                round(float(item.rotation.yaw), 4),
            ]
            for item in transforms
        ]
        route_sha256 = hashlib.sha256(
            json.dumps(serialized, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        identity = "|".join(
            (
                str(actor.type_id),
                self._actor_role(actor),
                f"{float(actor_transform.location.x):.4f}",
                f"{float(actor_transform.location.y):.4f}",
                f"{float(actor_transform.rotation.yaw):.3f}",
            )
        )
        return {
            "transforms": transforms,
            "cumulative_distance_m": cumulative,
            "route_sha256": route_sha256,
            "replay_identity": identity,
            "tick_index": 0,
            "last_distance_m": 0.0,
        }

    @staticmethod
    def _sample_replay_transform(
        transforms: Sequence[carla.Transform],
        cumulative_distance_m: Sequence[float],
        distance_m: float,
    ) -> carla.Transform:
        if not transforms or len(transforms) != len(cumulative_distance_m):
            raise ValueError("invalid native-lane replay trace")
        if float(distance_m) > float(cumulative_distance_m[-1]) + 1e-9:
            raise RuntimeError("native-lane replay exhausted its audited horizon")
        index = max(
            0,
            min(
                len(transforms) - 2,
                bisect.bisect_right(cumulative_distance_m, float(distance_m)) - 1,
            ),
        )
        left = transforms[index]
        right = transforms[index + 1]
        start_m = float(cumulative_distance_m[index])
        end_m = float(cumulative_distance_m[index + 1])
        alpha = 0.0 if end_m <= start_m else (float(distance_m) - start_m) / (end_m - start_m)
        alpha = max(0.0, min(1.0, alpha))
        yaw_delta = (
            float(right.rotation.yaw) - float(left.rotation.yaw) + 180.0
        ) % 360.0 - 180.0
        return carla.Transform(
            carla.Location(
                x=float(left.location.x)
                + alpha * (float(right.location.x) - float(left.location.x)),
                y=float(left.location.y)
                + alpha * (float(right.location.y) - float(left.location.y)),
                z=float(left.location.z)
                + alpha * (float(right.location.z) - float(left.location.z)),
            ),
            carla.Rotation(
                pitch=float(left.rotation.pitch)
                + alpha * (float(right.rotation.pitch) - float(left.rotation.pitch)),
                yaw=float(left.rotation.yaw) + alpha * yaw_delta,
                roll=float(left.rotation.roll)
                + alpha * (float(right.rotation.roll) - float(left.rotation.roll)),
            ),
        )

    def _audit_deterministic_replay_clearance(
        self, blockers: Sequence[carla.Actor]
    ) -> Dict[str, object]:
        """Reject an unsafe immutable replay before the first capture frame."""

        if self._replay_speed_mps is None or self._replay_fixed_delta_seconds is None:
            raise RuntimeError("deterministic replay was not initialized")
        minimum_allowed = float(
            self.integration["npc_trace_replay_minimum_clearance_m"]
        )
        minimum_pair = float("inf")
        minimum_blocker = float("inf")
        actor_ids = sorted(self._replay_route_state)
        frame_count = int(self.integration["traffic_expected_frame_count"])
        for tick_index in range(1, frame_count + 1):
            distance_m = (
                tick_index
                * self._replay_fixed_delta_seconds
                * self._replay_speed_mps
            )
            locations = {
                actor_id: self._sample_replay_transform(
                    self._replay_route_state[actor_id]["transforms"],
                    self._replay_route_state[actor_id]["cumulative_distance_m"],
                    distance_m,
                ).location
                for actor_id in actor_ids
            }
            for left_index, left_id in enumerate(actor_ids):
                for right_id in actor_ids[left_index + 1 :]:
                    minimum_pair = min(
                        minimum_pair,
                        float(locations[left_id].distance(locations[right_id])),
                    )
                for blocker in blockers:
                    minimum_blocker = min(
                        minimum_blocker,
                        float(locations[left_id].distance(blocker.get_location())),
                    )
        failures = []
        if minimum_pair < minimum_allowed:
            failures.append("ambient_replay_pair_clearance")
        if minimum_blocker < minimum_allowed:
            failures.append("ambient_replay_static_occluder_clearance")
        result = {
            "pass": not failures,
            "basis": "full_horizon_center_distance_before_capture",
            "minimum_allowed_clearance_m": minimum_allowed,
            "minimum_pair_clearance_m": (
                None if math.isinf(minimum_pair) else minimum_pair
            ),
            "minimum_static_occluder_clearance_m": (
                None if math.isinf(minimum_blocker) else minimum_blocker
            ),
            "failures": failures,
        }
        if failures:
            raise RuntimeError(
                f"deterministic ambient replay clearance failed: {result}"
            )
        return result

    def start(self) -> None:
        vehicles = self._npc_vehicles()
        ambient_walkers = [
            actor
            for actor in self.world.get_actors().filter("walker.pedestrian.*")
            if not self._actor_role(actor).startswith("phase2_")
            and not self._actor_role(actor).startswith("scenesense_")
        ]
        self._ambient_walker_ids = [int(actor.id) for actor in ambient_walkers]
        self._ambient_walker_metadata = {}
        ambient_walker_mode = str(
            self.integration.get(
                "ambient_walker_motion_mode", "walker_ai_destination"
            )
        )
        walker_group_ordinals: Dict[tuple[str, str], int] = {}
        ordered_walkers = sorted(
            ambient_walkers,
            key=lambda actor: (
                str(actor.type_id),
                self._actor_role(actor),
                float(actor.get_location().x),
                float(actor.get_location().y),
            ),
        )
        for actor in ordered_walkers:
            transform = actor.get_transform()
            key = (str(actor.type_id), self._actor_role(actor))
            ordinal = walker_group_ordinals.get(key, 0)
            walker_group_ordinals[key] = ordinal + 1
            identity = (
                f"stationary_context|{key[0]}|{key[1]}|{ordinal}"
                if ambient_walker_mode == "runner_owned_stationary"
                else "|".join(
                    (
                        key[0],
                        key[1],
                        f"{float(transform.location.x):.4f}",
                        f"{float(transform.location.y):.4f}",
                        f"{float(transform.rotation.yaw):.3f}",
                    )
                )
            )
            self._ambient_walker_metadata[int(actor.id)] = {
                "role_name": self._actor_role(actor),
                "type_id": str(actor.type_id),
                "replay_identity": identity,
                "replay_plan_sha256": hashlib.sha256(
                    f"{ambient_walker_mode}|{identity}".encode("utf-8")
                ).hexdigest(),
                "motion_mode": ambient_walker_mode,
            }
        blockers = [
            actor
            for actor in self.world.get_actors().filter("vehicle.*")
            if self._actor_role(actor).startswith("static_blocker_v4")
            or "_occluder" in self._actor_role(actor)
        ]
        self.actor_ids = [int(actor.id) for actor in vehicles]
        self.actor_metadata = {
            int(actor.id): {
                "role_name": self._actor_role(actor),
                "type_id": str(actor.type_id),
                "monitoring_scope": "ambient_npc",
            }
            for actor in vehicles
        }
        leading_distance = float(self.integration["tm_distance_to_leading_vehicle_m"])
        speed_difference = float(self.integration["tm_speed_difference_pct"])
        desired_speed = float(self.integration["tm_desired_speed_mps"])
        route_mode = str(self.integration.get("npc_route_mode", "fixed_loop"))
        self._route_mode = route_mode
        if route_mode not in {
            "fixed_loop",
            "tm_autonomous",
            "direct_loop",
            "deterministic_trace_replay",
            "stationary_context",
        }:
            raise ValueError(f"unsupported NPC route mode: {route_mode}")
        if route_mode == "direct_loop":
            self._direct_route_speed_mps = float(
                self.integration["npc_direct_route_speed_mps"]
            )
            if not 2.0 <= self._direct_route_speed_mps <= 10.0:
                raise ValueError("direct NPC route speed must be within 2-10 m/s")
        elif route_mode == "deterministic_trace_replay":
            self._direct_route_speed_mps = None
            self._replay_speed_mps = float(
                self.integration["npc_trace_replay_speed_mps"]
            )
            self._replay_fixed_delta_seconds = float(
                self.integration["npc_trace_replay_fixed_delta_seconds"]
            )
            horizon_m = float(self.integration["npc_trace_replay_horizon_m"])
            required_m = (
                self._replay_speed_mps
                * self._replay_fixed_delta_seconds
                * int(self.integration["traffic_expected_frame_count"])
            )
            if not 2.0 <= self._replay_speed_mps <= 10.0:
                raise ValueError("trace-replay speed must be within 2-10 m/s")
            if horizon_m < required_m + 10.0:
                raise ValueError(
                    "trace-replay horizon must retain at least 10 m after capture"
                )
        else:
            self._direct_route_speed_mps = None
            self._replay_speed_mps = None
            self._replay_fixed_delta_seconds = None
        loop_routes = (
            self._npc_loop_routes()
            if route_mode in {"fixed_loop", "direct_loop"}
            else []
        )
        loop_repetitions = int(self.integration.get("npc_loop_repetitions", 1))
        self.traffic_manager.set_global_distance_to_leading_vehicle(leading_distance)
        self.traffic_manager.global_percentage_speed_difference(speed_difference)
        stationary_identity_by_actor: Dict[int, str] = {}
        if route_mode == "stationary_context":
            group_ordinals: Dict[tuple[str, str], int] = {}
            ordered = sorted(
                vehicles,
                key=lambda actor: (
                    str(actor.type_id),
                    self._actor_role(actor),
                    float(actor.get_location().x),
                    float(actor.get_location().y),
                ),
            )
            for actor in ordered:
                key = (str(actor.type_id), self._actor_role(actor))
                ordinal = group_ordinals.get(key, 0)
                group_ordinals[key] = ordinal + 1
                stationary_identity_by_actor[int(actor.id)] = (
                    f"stationary_context|{key[0]}|{key[1]}|{ordinal}"
                )
        tm_failures = []
        for actor in vehicles:
            try:
                loop_route = (
                    self._nearest_loop_route(actor.get_location(), loop_routes)
                    if loop_routes
                    else []
                )
                route_priority_index = next(
                    (
                        index
                        for index, candidate in enumerate(loop_routes)
                        if candidate is loop_route
                    ),
                    0,
                )
                route_closed = bool(
                    loop_route and self._route_is_closed(loop_route)
                )
                self.traffic_manager.distance_to_leading_vehicle(actor, leading_distance)
                self.traffic_manager.vehicle_percentage_speed_difference(
                    actor, speed_difference
                )
                # CARLA Traffic Manager's API uses km/h even though our
                # experiment contract is expressed in SI units.
                self.traffic_manager.set_desired_speed(actor, desired_speed * 3.6)
                self.traffic_manager.auto_lane_change(actor, False)
                if route_mode == "fixed_loop":
                    if not route_closed:
                        raise RuntimeError(
                            "fixed_loop requires a geometrically closed NPC route"
                        )
                    self.traffic_manager.set_path(
                        actor,
                        self._rotated_loop_path(actor, loop_route, loop_repetitions),
                    )
                elif route_mode == "direct_loop":
                    actor.set_autopilot(False, int(self.integration["tm_port"]))
                    actor_location = actor.get_location()
                    waypoint_index = min(
                        range(len(loop_route)),
                        key=lambda index: actor_location.distance(loop_route[index]),
                    )
                    self._direct_route_state[int(actor.id)] = {
                        "waypoint_index": int(waypoint_index),
                        "route": loop_route,
                        "route_priority_index": int(route_priority_index),
                        "route_closed": route_closed,
                        "endpoint_reached": False,
                    }
                elif route_mode == "deterministic_trace_replay":
                    actor.set_autopilot(False, int(self.integration["tm_port"]))
                    actor.set_simulate_physics(False)
                    actor.set_target_velocity(carla.Vector3D())
                    actor.set_target_angular_velocity(carla.Vector3D())
                    self._replay_route_state[int(actor.id)] = (
                        self._build_native_lane_replay(
                            actor,
                            horizon_m=float(
                                self.integration["npc_trace_replay_horizon_m"]
                            ),
                            step_m=float(
                                self.integration["npc_trace_replay_step_m"]
                            ),
                        )
                    )
                    actor.set_transform(
                        self._replay_route_state[int(actor.id)]["transforms"][0]
                    )
                elif route_mode == "stationary_context":
                    actor.set_autopilot(False, int(self.integration["tm_port"]))
                    actor.set_simulate_physics(False)
                    actor.set_target_velocity(carla.Vector3D())
                    actor.set_target_angular_velocity(carla.Vector3D())
                    transform = actor.get_transform()
                    identity = stationary_identity_by_actor[int(actor.id)]
                    self._stationary_context_state[int(actor.id)] = {
                        "replay_identity": identity,
                        "plan_sha256": hashlib.sha256(
                            f"stationary_context|{identity}".encode("utf-8")
                        ).hexdigest(),
                        "tick_index": 0,
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

        replay_clearance = (
            self._audit_deterministic_replay_clearance(blockers)
            if route_mode == "deterministic_trace_replay"
            else None
        )

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
            "ambient_walker_count": int(len(ambient_walkers)),
            "ambient_walker_motion_mode": ambient_walker_mode,
            "static_blocker_count": int(len(blockers)),
            "minimum_npc_pairwise_distance_m": self._minimum_pairwise_distance(vehicles),
            "minimum_npc_to_static_blocker_distance_m": minimum_blocker_distance,
            "tm_distance_to_leading_vehicle_m": leading_distance,
            "tm_speed_difference_pct": speed_difference,
            "tm_desired_speed_mps": desired_speed,
            "npc_route_mode": route_mode,
            "npc_loop_route_count": int(len(loop_routes)),
            "npc_loop_route_points": int(sum(len(route) for route in loop_routes)),
            "npc_loop_route_point_counts": [int(len(route)) for route in loop_routes],
            "npc_route_closed": [
                bool(self._route_is_closed(route)) for route in loop_routes
            ],
            "npc_loop_repetitions": (
                int(loop_repetitions)
                if route_mode in {"fixed_loop", "direct_loop"}
                else None
            ),
            "deterministic_replay": (
                {
                    "speed_mps": self._replay_speed_mps,
                    "fixed_delta_seconds": self._replay_fixed_delta_seconds,
                    "horizon_m": float(
                        self.integration["npc_trace_replay_horizon_m"]
                    ),
                    "minimum_route_margin_m": min(
                        float(state["cumulative_distance_m"][-1])
                        - self._replay_speed_mps
                        * self._replay_fixed_delta_seconds
                        * int(self.integration["traffic_expected_frame_count"])
                        for state in self._replay_route_state.values()
                    ),
                    "route_sha256_by_replay_identity": {
                        str(state["replay_identity"]): str(state["route_sha256"])
                        for state in self._replay_route_state.values()
                    },
                    "clearance_gate": replay_clearance,
                }
                if route_mode == "deterministic_trace_replay"
                else None
            ),
            "stationary_context": (
                {
                    "actor_count": len(self._stationary_context_state),
                    "basis": "physics_disabled_immutable_ambient_context",
                    "gridlock_gate_applicable": False,
                }
                if route_mode == "stationary_context"
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
        externally_clocked = bool(self.integration.get("external_sync_tick_owner", False))
        self.initial_geometry["sampling_clock"] = (
            "external_sync_tick_owner" if externally_clocked else "carla_on_tick_callback"
        )
        if not externally_clocked:
            self._tick_callback_id = self.world.on_tick(self._on_tick)

    def activate_vehicle_motion(self, client: object | None = None) -> None:
        """Activate Traffic Manager after the held-population RELEASE barrier.

        The actor RPC has no acknowledgement payload.  Full-stack collection
        therefore uses an explicit command batch and rejects any per-actor
        registration error before advancing the first release tick.
        """

        if self._route_mode == "deterministic_trace_replay":
            self.initial_geometry["tm_activation"] = {
                "basis": "not_used_runner_owned_deterministic_trace_replay",
                "actor_count": len(self.actor_ids),
            }
            return
        if self._route_mode == "stationary_context":
            self.initial_geometry["tm_activation"] = {
                "basis": "not_used_runner_owned_stationary_context",
                "actor_count": len(self.actor_ids),
            }
            return
        if self._route_mode != "tm_autonomous":
            return
        live = self._live_vehicle_map()
        missing = sorted(set(self.actor_ids) - set(live))
        if missing:
            raise RuntimeError(
                f"ambient NPCs disappeared before Traffic Manager activation: {missing}"
            )
        tm_port = int(self.integration["tm_port"])
        if client is None:
            for actor_id in self.actor_ids:
                live[actor_id].set_autopilot(True, tm_port)
            activation = {
                "basis": "actor_rpc_unacknowledged_compatibility_path",
                "actor_count": len(self.actor_ids),
            }
        else:
            commands = [
                carla.command.SetAutopilot(int(actor_id), True, tm_port)
                for actor_id in self.actor_ids
            ]
            responses = list(client.apply_batch_sync(commands, False))
            failures = [
                {
                    "actor_id": int(self.actor_ids[index]),
                    "error": str(getattr(response, "error", "unknown error")),
                }
                for index, response in enumerate(responses)
                if str(getattr(response, "error", "")).strip()
            ]
            if len(responses) != len(commands):
                failures.append(
                    {
                        "actor_id": None,
                        "error": (
                            "Traffic Manager activation response count mismatch: "
                            f"expected {len(commands)}, observed {len(responses)}"
                        ),
                    }
                )
            if failures:
                raise RuntimeError(
                    f"Traffic Manager activation batch failed: {failures}"
                )
            activation = {
                "basis": "acknowledged_set_autopilot_batch",
                "actor_count": len(self.actor_ids),
                "tm_port": tm_port,
            }
        self.initial_geometry["tm_activation"] = activation

    def _remember_tick_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._tick_failure is None:
                self._tick_failure = f"{type(exc).__name__}: {exc}"

    def before_world_tick(self) -> None:
        """Apply direct-route control once before an externally owned tick."""

        try:
            if getattr(self, "_replay_route_state", {}):
                self._apply_deterministic_trace_replay()
            elif self._direct_route_state:
                self._apply_direct_route_controls()
        except BaseException as exc:
            self._remember_tick_failure(exc)

    def observe_snapshot(self, snapshot: object) -> None:
        """Record every realized synchronous frame after its world tick."""

        try:
            rows = []
            ambient_rows = []
            self._observed_tick_count = int(
                getattr(self, "_observed_tick_count", 0)
            ) + 1
            timestamp = float(snapshot.timestamp.elapsed_seconds)
            live_vehicles = self._live_vehicle_map()
            missing = sorted(set(self.actor_ids) - set(live_vehicles))
            if missing:
                raise RuntimeError(
                    "ambient NPC membership disappeared from the authoritative "
                    f"world inventory at frame {int(snapshot.frame)}: {missing}"
                )
            inferred_yields = self._infer_registered_hazard_yield_actor_ids(
                live_vehicles
            )
            for actor_id in self.actor_ids:
                actor_snapshot = snapshot.find(int(actor_id))
                replay_state = getattr(self, "_replay_route_state", {}).get(actor_id)
                stationary_state = getattr(
                    self, "_stationary_context_state", {}
                ).get(actor_id)
                if replay_state is not None:
                    actor = live_vehicles[int(actor_id)]
                    transform = actor.get_transform()
                    velocity = carla.Vector3D()
                    sample_source = "deterministic_trace_replay"
                elif stationary_state is not None:
                    actor = live_vehicles[int(actor_id)]
                    transform = actor.get_transform()
                    velocity = carla.Vector3D()
                    stationary_state["tick_index"] = self._observed_tick_count
                    sample_source = "stationary_context"
                elif actor_snapshot is None:
                    # CARLA 0.10 may omit live actors from WorldSnapshot after
                    # the first externally owned tick.  The world has already
                    # advanced synchronously, so an actor RPC is a causal,
                    # same-frame fallback rather than an additional tick.
                    actor = live_vehicles[int(actor_id)]
                    transform = actor.get_transform()
                    velocity = actor.get_velocity()
                    sample_source = "live_actor_fallback"
                else:
                    transform = actor_snapshot.get_transform()
                    velocity = actor_snapshot.get_velocity()
                    sample_source = "world_snapshot"
                metadata = self.actor_metadata[actor_id]
                row = {
                        "frame_id": int(snapshot.frame),
                        "carla_timestamp": timestamp,
                        "actor_id": int(actor_id),
                        "role_name": metadata["role_name"],
                        "type_id": metadata["type_id"],
                        "world_x": float(transform.location.x),
                        "world_y": float(transform.location.y),
                        "world_z": float(transform.location.z),
                        "sample_source": sample_source,
                        "speed_mps": (
                            float(self._replay_speed_mps)
                            if replay_state is not None
                            else 0.0
                            if stationary_state is not None
                            else float(
                                math.sqrt(
                                    velocity.x ** 2
                                    + velocity.y ** 2
                                    + velocity.z ** 2
                                )
                            )
                        ),
                        "replay_identity": (
                            str(replay_state["replay_identity"])
                            if replay_state is not None
                            else str(stationary_state["replay_identity"])
                            if stationary_state is not None
                            else ""
                        ),
                        "replay_plan_sha256": (
                            str(replay_state["route_sha256"])
                            if replay_state is not None
                            else str(stationary_state["plan_sha256"])
                            if stationary_state is not None
                            else ""
                        ),
                        "replay_tick_index": (
                            int(replay_state["tick_index"])
                            if replay_state is not None
                            else int(stationary_state["tick_index"])
                            if stationary_state is not None
                            else None
                        ),
                        "replay_distance_m": (
                            float(replay_state["last_distance_m"])
                            if replay_state is not None
                            else 0.0
                            if stationary_state is not None
                            else None
                        ),
                        "registered_hazard_yield_active": bool(
                            actor_id in inferred_yields
                            or actor_id in self._registered_hazard_yield_actor_ids
                        ),
                    }
                rows.append(row)
                ambient_rows.append({**row, "actor_kind": "vehicle"})
            ambient_walker_ids = list(
                getattr(self, "_ambient_walker_ids", [])
            )
            live_walkers = (
                {
                    int(actor.id): actor
                    for actor in self.world.get_actors().filter(
                        "walker.pedestrian.*"
                    )
                }
                if ambient_walker_ids
                else {}
            )
            missing_walkers = sorted(
                set(ambient_walker_ids) - set(live_walkers)
            )
            if missing_walkers:
                raise RuntimeError(
                    "ambient walkers disappeared from the authoritative world "
                    f"inventory at frame {int(snapshot.frame)}: {missing_walkers}"
                )
            for actor_id in ambient_walker_ids:
                actor = live_walkers[actor_id]
                actor_snapshot = snapshot.find(int(actor_id))
                if actor_snapshot is None:
                    transform = actor.get_transform()
                    velocity = actor.get_velocity()
                    sample_source = "live_actor_fallback"
                else:
                    transform = actor_snapshot.get_transform()
                    velocity = actor_snapshot.get_velocity()
                    sample_source = "world_snapshot"
                metadata = self._ambient_walker_metadata[actor_id]
                ambient_rows.append(
                    {
                        "frame_id": int(snapshot.frame),
                        "carla_timestamp": timestamp,
                        "actor_id": int(actor_id),
                        "role_name": metadata["role_name"],
                        "type_id": metadata["type_id"],
                        "world_x": float(transform.location.x),
                        "world_y": float(transform.location.y),
                        "world_z": float(transform.location.z),
                        "sample_source": sample_source,
                        "speed_mps": float(
                            math.sqrt(
                                velocity.x ** 2
                                + velocity.y ** 2
                                + velocity.z ** 2
                            )
                        ),
                        "replay_identity": metadata["replay_identity"],
                        "replay_plan_sha256": metadata["replay_plan_sha256"],
                        "replay_tick_index": self._observed_tick_count,
                        "replay_distance_m": 0.0,
                        "registered_hazard_yield_active": False,
                        "actor_kind": "walker",
                    }
                )
            with self._lock:
                self.trajectory_rows.extend(rows)
                if not hasattr(self, "ambient_actor_trajectory_rows"):
                    self.ambient_actor_trajectory_rows = []
                self.ambient_actor_trajectory_rows.extend(ambient_rows)
        except BaseException as exc:
            self._remember_tick_failure(exc)

    def _on_tick(self, snapshot: object) -> None:
        self.before_world_tick()
        self.observe_snapshot(snapshot)

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._tick_failure
        if failure is not None:
            raise RuntimeError(f"NPC traffic tick callback failed: {failure}")

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _infer_registered_hazard_yield_actor_ids(
        self, live_vehicles: Mapping[int, carla.Actor]
    ) -> set[int]:
        """Attribute a yield only to NPCs whose forward envelope holds the hazard."""

        registered_vehicles = [
            actor
            for actor in live_vehicles.values()
            if self._actor_role(actor).startswith("phase2_registered_target_")
        ]
        registered_walkers = [
            actor
            for actor in self.world.get_actors().filter("walker.pedestrian.*")
            if self._actor_role(actor).startswith("phase2_registered_target_")
        ]
        yielding: set[int] = set()
        for actor_id in self.actor_ids:
            actor = live_vehicles.get(actor_id)
            if actor is None:
                continue
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            speed_mps = math.sqrt(
                float(velocity.x) ** 2
                + float(velocity.y) ** 2
                + float(velocity.z) ** 2
            )
            if any(
                _vehicle_envelope_requires_yield(
                    actor,
                    transform,
                    other,
                    speed_mps=speed_mps,
                    maximum_forward_m=12.0,
                )
                for other in registered_vehicles
                if int(other.id) != actor_id
            ):
                yielding.add(actor_id)
                continue
            for walker in registered_walkers:
                try:
                    walker_location = walker.get_location()
                    walker_velocity = walker.get_velocity()
                except RuntimeError:
                    continue
                if _walker_requires_yield(
                    transform,
                    walker_location,
                    walker_velocity,
                    registered_crossing=True,
                ):
                    yielding.add(actor_id)
                    break
        return yielding

    def _apply_direct_route_controls(self) -> None:
        """Apply bounded synchronous waypoint control to all managed NPCs."""

        self._registered_hazard_yield_actor_ids = set()

        all_vehicles = self._live_vehicle_map()
        actors = {
            actor_id: all_vehicles[actor_id]
            for actor_id in self.actor_ids
            if actor_id in all_vehicles
        }
        missing = sorted(set(self.actor_ids) - set(actors))
        if missing:
            raise RuntimeError(f"ambient NPCs disappeared before direct control: {missing}")
        all_walkers = list(self.world.get_actors().filter("walker.pedestrian.*"))
        if self._direct_route_speed_mps is None:
            raise RuntimeError("direct-route controller was not initialized")
        target_speed = self._direct_route_speed_mps
        for actor_id, state in self._direct_route_state.items():
            actor = actors.get(actor_id)
            if actor is None:
                continue
            route = state["route"]
            index = int(state["waypoint_index"])
            route_closed = bool(state.get("route_closed", False))
            transform = actor.get_transform()
            location = transform.location
            for _unused in range(len(route)):
                target = route[index]
                if location.distance(target) >= 4.0:
                    break
                if index + 1 < len(route):
                    index += 1
                elif route_closed:
                    index = 0
                else:
                    state["endpoint_reached"] = True
                    break
            if bool(state.get("endpoint_reached", False)):
                actor.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                    )
                )
                state["waypoint_index"] = int(index)
                continue
            if index + 1 < len(route):
                lookahead = route[index + 1]
            elif route_closed:
                lookahead = route[0]
            else:
                lookahead = route[index]
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
                if _vehicle_envelope_requires_yield(
                    actor,
                    transform,
                    other,
                    speed_mps=speed_mps,
                    maximum_forward_m=12.0,
                ):
                    other_state = self._direct_route_state.get(other_id)
                    if (
                        other_state is not None
                        and int(other_state["route_priority_index"])
                        != int(state["route_priority_index"])
                        and int(state["route_priority_index"])
                        < int(other_state["route_priority_index"])
                    ):
                        # Reviewed ambient paths are ordered helper/through
                        # first, recipient/turning second. Only the lower-
                        # priority route yields at a crossing; otherwise two
                        # symmetric shields can stop nose-to-nose forever.
                        continue
                    if self._actor_role(other).startswith(
                        "phase2_registered_target_"
                    ):
                        self._registered_hazard_yield_actor_ids.add(actor_id)
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
                    registered_crossing = self._actor_role(walker).startswith(
                        "phase2_registered_target_"
                    )
                    if _walker_requires_yield(
                        transform,
                        walker_location,
                        walker_velocity,
                        registered_crossing=registered_crossing,
                    ):
                        if registered_crossing:
                            self._registered_hazard_yield_actor_ids.add(actor_id)
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

    def _apply_deterministic_trace_replay(self) -> None:
        """Advance every ambient NPC on its immutable native-lane trace."""

        if self._replay_speed_mps is None or self._replay_fixed_delta_seconds is None:
            raise RuntimeError("deterministic replay was not initialized")
        live = self._live_vehicle_map()
        missing = sorted(set(self.actor_ids) - set(live))
        if missing:
            raise RuntimeError(
                f"ambient NPCs disappeared before deterministic replay: {missing}"
            )
        for actor_id, state in self._replay_route_state.items():
            tick_index = int(state["tick_index"]) + 1
            distance_m = (
                tick_index
                * self._replay_fixed_delta_seconds
                * self._replay_speed_mps
            )
            transform = self._sample_replay_transform(
                state["transforms"],
                state["cumulative_distance_m"],
                distance_m,
            )
            actor = live[actor_id]
            actor.set_autopilot(False, int(self.integration["tm_port"]))
            actor.set_simulate_physics(False)
            actor.set_target_velocity(carla.Vector3D())
            actor.set_target_angular_velocity(carla.Vector3D())
            actor.set_transform(transform)
            state["tick_index"] = tick_index
            state["last_distance_m"] = distance_m

    def _on_collision(self, npc_actor_id: int, event: object) -> None:
        other = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        row = {
            "frame_id": int(getattr(event, "frame", -1)),
            "carla_timestamp": float(getattr(event, "timestamp", float("nan"))),
            "npc_actor_id": int(npc_actor_id),
            "npc_role_name": self.actor_metadata.get(npc_actor_id, {}).get("role_name", ""),
            "contact_owner_scope": self.actor_metadata.get(npc_actor_id, {}).get(
                "monitoring_scope", "unknown"
            ),
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
            ambient_actor_rows = list(
                getattr(self, "ambient_actor_trajectory_rows", [])
            )
            collision_rows = list(self.collision_rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_fields = (
            "frame_id", "carla_timestamp", "actor_id", "role_name", "type_id",
            "world_x", "world_y", "world_z", "sample_source", "speed_mps",
            "replay_identity", "replay_plan_sha256", "replay_tick_index",
            "replay_distance_m",
            "registered_hazard_yield_active",
        )
        collision_fields = (
            "frame_id", "carla_timestamp", "npc_actor_id", "npc_role_name",
            "contact_owner_scope",
            "other_actor_id", "other_type_id", "other_role_name",
            "normal_impulse_x", "normal_impulse_y", "normal_impulse_z",
        )
        self._write_csv(self.output_dir / "npc_trajectories.csv", trajectory_rows, trajectory_fields)
        ambient_trajectory_fields = (*trajectory_fields, "actor_kind")
        self._write_csv(
            self.output_dir / "ambient_actor_trajectories.csv",
            ambient_actor_rows,
            ambient_trajectory_fields,
        )
        self._write_csv(self.output_dir / "npc_collision_events.csv", collision_rows, collision_fields)
        summary = _traffic_sanity_summary(
            pd.DataFrame(trajectory_rows, columns=trajectory_fields),
            pd.DataFrame(collision_rows, columns=collision_fields),
            self.actor_ids,
            self.integration["traffic_sanity_gate"],
            (
                int(self.integration["traffic_expected_frame_count"])
                if self.integration.get("traffic_expected_frame_count") is not None
                else None
            ),
            stationary_context_expected=bool(
                self.integration.get("expected_stationary_context", False)
            ),
            stationary_context_trajectories=pd.DataFrame(
                ambient_actor_rows, columns=ambient_trajectory_fields
            ),
        )
        summary["sample_source_counts"] = {
            str(key): int(value)
            for key, value in pd.Series(
                [row.get("sample_source", "unknown") for row in trajectory_rows],
                dtype=str,
            ).value_counts().items()
        }
        with self._lock:
            tick_failure = self._tick_failure
        summary["tick_callback_error"] = tick_failure
        if tick_failure is not None:
            summary["pass"] = False
            summary.setdefault("failures", []).append("npc_tick_callback_error")
        summary["initial_geometry"] = self.initial_geometry
        summary["trajectory_csv"] = str(
            self.output_dir / "npc_trajectories.csv"
        )
        summary["ambient_actor_trajectory_csv"] = str(
            self.output_dir / "ambient_actor_trajectories.csv"
        )
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

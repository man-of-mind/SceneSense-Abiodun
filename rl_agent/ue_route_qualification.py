"""Ego-only repeatable-route qualification for the UE experiment.

The module intentionally has no RGB, radar, perception-model, OAI, spatial-map,
NPC, pedestrian, or Traffic Manager path. CARLA is imported lazily; all route
logic and evidence contracts can therefore be tested offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "rl_agent/configs/ue_route_qualification_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "rl_agent/experiments/ue_route_qualification_v1"

CONFIG_SCHEMA = "scenesense.ue_route_qualification_config.v1"
MANIFEST_SCHEMA = "scenesense.ue_route_qualification_manifest.v1"
ROUTE_CONTRACT_SCHEMA = "scenesense.ue_route_contract.v1"
MACHINE_REVIEW_SCHEMA = "scenesense.ue_route_machine_review.v1"
MANUAL_REVIEW_SCHEMA = "scenesense.ue_route_manual_review.v1"
FINAL_REVIEW_SCHEMA = "scenesense.ue_route_final_review.v1"

FROZEN_EGO_BLUEPRINT = "vehicle.lincoln.mkz"
FROZEN_ROUTE_JSON_SHA256 = (
    "0d3cceeb30d603e258cc61c00bb51e8d0ca29c176e7fccb38ec1e10692233860"
)
FROZEN_PROGRESS_CSV_SHA256 = (
    "f3dc2f4d8c59905801fdfad2df7a19f2b427459d4039ed3a8cdec3535e818ce1"
)
REQUIRED_MANUAL_CHECKS = (
    "spawn_lane_heading_and_no_physics_jump",
    "expected_perimeter_and_all_intended_turns",
    "no_curb_sidewalk_wrong_lane_corner_cut_oscillation_reverse_or_u_turn",
    "stable_steering_acceleration_braking_and_turn_speed",
    "smooth_final_to_first_seam_without_teleport_or_abrupt_correction",
    "full_route_completed_before_return_counted",
    "trials_two_and_three_materially_match_trial_one",
    "previous_ego_and_collision_sensor_absent_before_next_trial",
)

TRACE_FIELDS = (
    "experiment_id",
    "trial_id",
    "frame_id",
    "sim_time_s",
    "elapsed_s",
    "sim_delta_s",
    "tick_wall_s",
    "pacing_sleep_s",
    "pacing_lateness_s",
    "future_decision_slot",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_yaw_deg",
    "ego_speed_mps",
    "route_index",
    "route_target_index",
    "unwrapped_progress_m",
    "lap_count",
    "wrap_count",
    "cross_track_m",
    "heading_error_deg",
    "return_position_error_m",
    "return_heading_error_deg",
    "throttle",
    "steer",
    "brake",
    "collision_count",
    "stall_s",
    "divergence_s",
    "lap_armed",
)

EVENT_FIELDS = (
    "experiment_id",
    "trial_id",
    "event_id",
    "event_type",
    "frame_id",
    "sim_time_s",
    "route_index",
    "unwrapped_progress_m",
    "status",
    "details",
)


class RouteQualificationError(RuntimeError):
    """Raised when a frozen route or evidence contract is invalid."""


class MonotonicTickPacer:
    """Hold synchronous ticks to a wall-clock period without changing sim gates.

    Deadlines advance from the original monotonic schedule. If one tick runs
    late, subsequent calls omit or shorten sleep until the schedule catches up;
    no wall-clock observation enters route acceptance.
    """

    def __init__(
        self,
        period_s: float,
        *,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if float(period_s) <= 0.0:
            raise ValueError("pacing period must be positive")
        self.period_s = float(period_s)
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._next_deadline_s = float(self._monotonic()) + self.period_s

    def wait(self) -> dict[str, float]:
        deadline = self._next_deadline_s
        now = float(self._monotonic())
        sleep_s = max(0.0, deadline - now)
        if sleep_s > 0.0:
            self._sleeper(sleep_s)
        observed = float(self._monotonic())
        lateness_s = max(0.0, observed - deadline)
        self._next_deadline_s = deadline + self.period_s
        return {
            "sleep_s": float(sleep_s),
            "lateness_s": float(lateness_s),
        }


@dataclass(frozen=True)
class RoutePoint:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Projection:
    cross_track_m: float
    along_route_m: float
    progress_fraction: float
    segment_index: int


@dataclass(frozen=True)
class ControlCommand:
    throttle: float
    steer: float
    brake: float
    route_index: int
    target_index: int
    target_x: float
    target_y: float
    heading_error_rad: float
    transitions: tuple[tuple[int, int], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def atomic_write_new_text(path: Path, text: str) -> None:
    """Atomically publish one create-only UTF-8 artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RouteQualificationError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_new_json(path: Path, value: object) -> None:
    atomic_write_new_text(path, json_text(value))


def rows_to_csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def wrap_radians(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def read_route_points(path: Path) -> list[RoutePoint]:
    points: list[RoutePoint] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            points.append(
                RoutePoint(
                    x=float(row["ego_x"]),
                    y=float(row["ego_y"]),
                    z=float(row.get("ego_z", 0.0)),
                )
            )
    if len(points) < 3:
        raise RouteQualificationError("closed route requires at least three points")
    return points


def open_route_length_m(points: Sequence[RoutePoint]) -> float:
    if len(points) < 2:
        raise RouteQualificationError("route requires at least two points")
    return float(
        sum(
            math.hypot(end.x - start.x, end.y - start.y)
            for start, end in zip(points, points[1:])
        )
    )


def closing_seam_length_m(points: Sequence[RoutePoint]) -> float:
    if len(points) < 2:
        raise RouteQualificationError("route requires at least two points")
    return float(math.hypot(points[0].x - points[-1].x, points[0].y - points[-1].y))


def closed_route_length_m(points: Sequence[RoutePoint]) -> float:
    return open_route_length_m(points) + closing_seam_length_m(points)


def project_to_closed_route(
    x_m: float, y_m: float, points: Sequence[RoutePoint]
) -> Projection:
    route_length = closed_route_length_m(points)
    best: tuple[float, float, int] | None = None
    cumulative_m = 0.0
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        fraction = (
            0.0
            if length_sq <= 1e-12
            else max(
                0.0,
                min(
                    1.0,
                    ((float(x_m) - start.x) * dx + (float(y_m) - start.y) * dy)
                    / length_sq,
                ),
            )
        )
        nearest_x = start.x + fraction * dx
        nearest_y = start.y + fraction * dy
        distance = math.hypot(float(x_m) - nearest_x, float(y_m) - nearest_y)
        length = math.sqrt(length_sq)
        candidate = (distance, cumulative_m + fraction * length, index)
        if best is None or candidate[0] < best[0]:
            best = candidate
        cumulative_m += length
    assert best is not None
    return Projection(
        cross_track_m=float(best[0]),
        along_route_m=float(best[1]),
        progress_fraction=float((best[1] / route_length) % 1.0),
        segment_index=int(best[2]),
    )


class ClosedDirectRouteController:
    """Deterministic closed-route kernel using the existing direct-control gains."""

    def __init__(self, points: Sequence[RoutePoint], config: Mapping[str, Any]) -> None:
        self.points = list(points)
        self.config = dict(config)
        self.index: int | None = None

    def reset(self, x_m: float, y_m: float) -> None:
        self.index = min(
            range(len(self.points)),
            key=lambda index: math.hypot(
                self.points[index].x - float(x_m),
                self.points[index].y - float(y_m),
            ),
        )
        # Normalize the seam-adjacent spawn without recording a lap wrap.
        for _unused in range(len(self.points)):
            point = self.points[self.index]
            if math.hypot(point.x - float(x_m), point.y - float(y_m)) >= float(
                self.config["waypoint_reach_m"]
            ):
                break
            self.index = (self.index + 1) % len(self.points)

    def command(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_deg: float,
        speed_mps: float,
        target_speed_mps: float,
    ) -> ControlCommand:
        if self.index is None:
            self.reset(x_m, y_m)
        assert self.index is not None
        transitions: list[tuple[int, int]] = []
        for _unused in range(len(self.points)):
            point = self.points[self.index]
            if math.hypot(point.x - float(x_m), point.y - float(y_m)) >= float(
                self.config["waypoint_reach_m"]
            ):
                break
            previous = self.index
            self.index = (self.index + 1) % len(self.points)
            transitions.append((previous, self.index))

        target_index = (
            self.index + int(self.config["lookahead_points"])
        ) % len(self.points)
        target = self.points[target_index]
        desired_yaw = math.atan2(target.y - float(y_m), target.x - float(x_m))
        heading_error = wrap_radians(desired_yaw - math.radians(float(yaw_deg)))
        turn_scale = max(
            float(self.config["turn_minimum_speed_scale"]),
            1.0 - abs(heading_error) / math.pi,
        )
        speed_error = float(target_speed_mps) * turn_scale - float(speed_mps)
        throttle = max(
            0.0,
            min(
                float(self.config["throttle_max"]),
                float(self.config["throttle_gain"]) * speed_error,
            ),
        )
        brake = max(
            0.0,
            min(
                float(self.config["brake_max"]),
                -float(self.config["brake_gain"]) * speed_error,
            ),
        )
        steer = max(
            -float(self.config["steer_max_abs"]),
            min(
                float(self.config["steer_max_abs"]),
                heading_error
                / math.radians(float(self.config["steer_full_scale_deg"])),
            ),
        )
        return ControlCommand(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            route_index=int(self.index),
            target_index=int(target_index),
            target_x=float(target.x),
            target_y=float(target.y),
            heading_error_rad=float(heading_error),
            transitions=tuple(transitions),
        )


class OrderedRouteProgress:
    """Prove sequential controller indices and exactly count the closing wrap."""

    def __init__(self, points: Sequence[RoutePoint], start_index: int) -> None:
        self.points = list(points)
        self.expected_index = int(start_index)
        self.unwrapped_progress_m = 0.0
        self.wrap_count = 0
        self.wrap_after_arming_count = 0
        self.sequence_error_count = 0

    def observe(self, transition: tuple[int, int], *, armed_before: bool) -> dict[str, Any]:
        previous, current = (int(transition[0]), int(transition[1]))
        expected_current = (previous + 1) % len(self.points)
        ordered = previous == self.expected_index and current == expected_current
        if not ordered:
            self.sequence_error_count += 1
        # Advancing previous->current means `previous` was reached. The first
        # reached point is point 0 from the seam-adjacent spawn, so add the
        # segment ending at `previous`.
        segment_start = self.points[(previous - 1) % len(self.points)]
        segment_end = self.points[previous]
        self.unwrapped_progress_m += math.hypot(
            segment_end.x - segment_start.x, segment_end.y - segment_start.y
        )
        wrapped = previous == len(self.points) - 1 and current == 0
        if wrapped:
            self.wrap_count += 1
            if armed_before:
                self.wrap_after_arming_count += 1
        self.expected_index = current
        return {
            "ordered": ordered,
            "wrapped": wrapped,
            "armed_before_wrap": bool(armed_before) if wrapped else None,
            "unwrapped_progress_m": float(self.unwrapped_progress_m),
        }


class ArmedLapDetector:
    """Count a return only after ordered 95% progress and one armed wrap."""

    def __init__(self, route_length_m: float, config: Mapping[str, Any]) -> None:
        self.route_length_m = float(route_length_m)
        self.config = dict(config)
        self.exited_start_gate = False
        self.armed = False
        self.completed = False
        self.lap_count = 0
        self.false_completion_count = 0
        self.previous_inside_completion_gate = True

    def update(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_deg: float,
        unwrapped_progress_m: float,
        wrap_count: int,
        wrap_after_arming_count: int,
    ) -> dict[str, Any]:
        gate_distance = math.hypot(
            float(x_m) - float(self.config["start_gate_x_m"]),
            float(y_m) - float(self.config["start_gate_y_m"]),
        )
        heading_error = abs(
            wrap_degrees(
                float(yaw_deg) - float(self.config["expected_return_heading_deg"])
            )
        )
        inside = gate_distance <= float(self.config["completion_radius_m"])
        if gate_distance >= float(self.config["gate_exit_radius_m"]):
            self.exited_start_gate = True
        if (
            self.exited_start_gate
            and float(unwrapped_progress_m)
            >= self.route_length_m
            * float(self.config["minimum_ordered_progress_ratio_to_arm"])
        ):
            self.armed = True

        valid_return = bool(
            self.armed
            and int(wrap_count) == int(self.config["required_wrap_count"])
            and int(wrap_after_arming_count) == int(self.config["required_wrap_count"])
            and inside
            and heading_error <= float(self.config["completion_heading_tolerance_deg"])
        )
        if valid_return and not self.completed:
            self.completed = True
            self.lap_count = 1
        elif (
            self.exited_start_gate
            and inside
            and not self.previous_inside_completion_gate
            and not self.armed
            and not valid_return
        ):
            # Only an early re-entry is a false completion. On the normal
            # final approach the 4 m completion gate is entered just before
            # the controller advances route index 84->0; an already-armed
            # approach must be allowed to continue to that wrap.
            self.false_completion_count += 1
        self.previous_inside_completion_gate = inside
        return {
            "start_gate_distance_m": float(gate_distance),
            "return_heading_error_deg": float(heading_error),
            "inside_completion_gate": bool(inside),
            "exited_start_gate": bool(self.exited_start_gate),
            "armed": bool(self.armed),
            "completed": bool(self.completed),
            "lap_count": int(self.lap_count),
            "false_completion_count": int(self.false_completion_count),
        }


class CollisionMailbox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def callback(self, event: Any) -> None:
        impulse = event.normal_impulse
        other = event.other_actor
        with self._lock:
            self._rows.append(
                {
                    "frame_id": int(event.frame),
                    "other_actor_id": int(other.id),
                    "other_actor_type": str(other.type_id),
                    "impulse": [
                        float(impulse.x),
                        float(impulse.y),
                        float(impulse.z),
                    ],
                    "impulse_norm": float(
                        math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
                    ),
                }
            )

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows]

    def count(self) -> int:
        with self._lock:
            return len(self._rows)


def flush_collision_tick(
    world: Any,
    mailbox: Any,
    *,
    tick_timeout_s: float,
    settle_s: float,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    """Advance one unmeasured tick while the collision sensor remains alive.

    The bounded settle lets a callback queued by that tick reach the mailbox.
    This helper never receives or mutates controller, route, lap, or trace
    state, so the flush cannot become part of the qualified lap.
    """

    before_rows = mailbox.rows()
    result: dict[str, Any] = {
        "pass": False,
        "flush_tick_count": 0,
        "flush_frame_id": None,
        "mailbox_count_before": len(before_rows),
        "mailbox_count_after": len(before_rows),
        "new_collision_rows": [],
        "error": None,
    }
    try:
        result["flush_frame_id"] = int(world.tick(float(tick_timeout_s)))
        result["flush_tick_count"] = 1
        if float(settle_s) > 0.0:
            sleeper(float(settle_s))
        after_rows = mailbox.rows()
        result["mailbox_count_after"] = len(after_rows)
        result["new_collision_rows"] = after_rows[len(before_rows) :]
        result["pass"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteQualificationError(f"{label} must be an object")
    return value


def select_exact_blueprint(blueprint_library: Any, blueprint_id: str) -> Any:
    """Resolve one exact CARLA blueprint with a readable missing-ID error."""

    requested = str(blueprint_id)
    candidates = [
        blueprint
        for blueprint in blueprint_library.filter(requested)
        if str(blueprint.id) == requested
    ]
    if len(candidates) != 1:
        available = sorted(
            str(blueprint.id)
            for blueprint in blueprint_library.filter("vehicle.*")
        )
        raise RouteQualificationError(
            "frozen ego blueprint is unavailable in this CARLA build: "
            f"requested={requested}, available_vehicle_blueprints={available}"
        )
    return candidates[0]


def load_and_validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema") != CONFIG_SCHEMA:
        raise RouteQualificationError(f"unsupported config schema: {config.get('schema')}")

    carla_config = _require_mapping(config.get("carla"), "carla")
    scene = _require_mapping(config.get("scene"), "scene")
    route = _require_mapping(config.get("route"), "route")
    clock = _require_mapping(config.get("clock"), "clock")
    trials = _require_mapping(config.get("trials"), "trials")
    acceptance = _require_mapping(config.get("acceptance"), "acceptance")
    lap = _require_mapping(route.get("lap_detector"), "route.lap_detector")
    manual = _require_mapping(config.get("manual_review"), "manual_review")

    if str(scene.get("contract")) != "ego_only":
        raise RouteQualificationError("qualification scene must be ego_only")
    forbidden = (
        "rgb_enabled",
        "radar_enabled",
        "model_enabled",
        "oai_enabled",
        "map_stream_enabled",
        "traffic_manager_enabled",
    )
    if any(bool(scene.get(name)) for name in forbidden):
        raise RouteQualificationError("RGB/radar/model/OAI/map/TM must remain disabled")
    if not bool(clock.get("synchronous_mode")):
        raise RouteQualificationError("qualification requires synchronous mode")
    if abs(float(clock["control_hz"]) - 20.0) > 1e-9 or abs(
        float(clock["control_hz"]) * float(clock["fixed_delta_seconds"]) - 1.0
    ) > 1e-9:
        raise RouteQualificationError("frozen clock must be 20 Hz / 0.05 s")
    if not (
        bool(clock["real_time_pacing_enabled"])
        and float(clock["real_time_tick_period_s"]) == 0.05
        and float(clock["real_time_tick_period_s"])
        == float(clock["fixed_delta_seconds"])
        and int(clock["collision_flush_ticks"]) == 1
        and float(clock["collision_flush_settle_s"]) == 0.02
    ):
        raise RouteQualificationError(
            "frozen real-time pacing or one-tick collision flush contract drifted"
        )
    if int(trials["count"]) != 3:
        raise RouteQualificationError("qualification requires exactly three trials")
    if int(route["spawn_index"]) != 55 or float(route["target_speed_mps"]) != 6.0:
        raise RouteQualificationError("frozen spawn/speed must be index 55 and 6 m/s")
    if str(route["ego_blueprint"]) != FROZEN_EGO_BLUEPRINT or tuple(
        scene.get("allowed_spawned_actor_types", [])
    ) != (FROZEN_EGO_BLUEPRINT, "sensor.other.collision"):
        raise RouteQualificationError(
            "frozen CARLA 0.10 ego/collision actor contract drifted"
        )
    if float(lap["minimum_ordered_progress_ratio_to_arm"]) != 0.95:
        raise RouteQualificationError("lap detector must arm at exactly 95% progress")
    if int(lap["required_wrap_count"]) != 1:
        raise RouteQualificationError("lap detector must require exactly one wrap")
    if float(lap["completion_radius_m"]) != 4.0:
        raise RouteQualificationError("completion radius must be 4 m")
    if tuple(manual.get("required_checks", [])) != REQUIRED_MANUAL_CHECKS:
        raise RouteQualificationError("manual review must contain the exact eight checks")
    if str(route["route_json_sha256"]) != FROZEN_ROUTE_JSON_SHA256 or str(
        route["progress_csv_sha256"]
    ) != FROZEN_PROGRESS_CSV_SHA256:
        raise RouteQualificationError("frozen route hash declarations drifted")
    if not (
        float(trials["maximum_duration_s"]) == 120.0
        and float(lap["completion_heading_tolerance_deg"]) == 15.0
        and int(acceptance["maximum_collision_count"]) == 0
        and float(acceptance["stall_speed_threshold_mps"]) == 0.5
        and float(acceptance["maximum_continuous_stall_s"]) == 5.0
        and float(acceptance["maximum_duration_spread_fraction_of_median"]) == 0.05
        and bool(acceptance["require_owned_actor_cleanup"])
        and float(acceptance["cleanup_max_sim_s"]) == 5.0
    ):
        raise RouteQualificationError("frozen duration/collision/stall gates drifted")

    route_json = resolve_repo_path(str(route["route_json"]))
    progress_csv = resolve_repo_path(str(route["progress_csv"]))
    for label, candidate, expected_hash in (
        ("route JSON", route_json, str(route["route_json_sha256"])),
        ("progress CSV", progress_csv, str(route["progress_csv_sha256"])),
    ):
        if not candidate.is_file():
            raise RouteQualificationError(f"{label} is missing: {candidate}")
        observed_hash = sha256_file(candidate)
        if observed_hash != expected_hash:
            raise RouteQualificationError(
                f"{label} hash mismatch: expected={expected_hash}, observed={observed_hash}"
            )
    points = read_route_points(progress_csv)
    with route_json.open("r", encoding="utf-8") as stream:
        route_metadata = json.load(stream)
    if str(route_metadata.get("map")) != "Carla/Maps/Town10HD_Opt":
        raise RouteQualificationError("route JSON map asset drifted")
    if len(points) != int(route["expected_point_count"]):
        raise RouteQualificationError("route point-count drift")
    open_length = open_route_length_m(points)
    seam_length = closing_seam_length_m(points)
    closed_length = open_length + seam_length
    tolerance = float(route["length_tolerance_m"])
    for label, observed, expected in (
        ("open route", open_length, float(route["expected_open_length_m"])),
        ("closing seam", seam_length, float(route["expected_closing_seam_m"])),
        ("closed route", closed_length, float(route["expected_closed_length_m"])),
    ):
        if abs(observed - expected) > tolerance:
            raise RouteQualificationError(
                f"{label} length drift: expected={expected:.3f}, observed={observed:.3f}"
            )
    if not (
        float(acceptance["minimum_unwrapped_progress_ratio"]) == 0.95
        and float(acceptance["maximum_unwrapped_progress_ratio"]) == 1.10
        and float(acceptance["maximum_cross_track_p95_m"]) == 1.5
        and float(acceptance["maximum_continuous_divergence_s"]) == 0.5
        and float(acceptance["maximum_cross_track_error_m"]) == 3.0
    ):
        raise RouteQualificationError("frozen progress/cross-track gates drifted")

    resolved = {
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "route_json_path": str(route_json),
        "route_json_sha256": sha256_file(route_json),
        "progress_csv_path": str(progress_csv),
        "progress_csv_sha256": sha256_file(progress_csv),
        "route_point_count": len(points),
        "open_route_length_m": open_length,
        "closing_seam_m": seam_length,
        "closed_route_length_m": closed_length,
        "ideal_duration_at_target_speed_s": closed_length / float(route["target_speed_mps"]),
        "map": str(carla_config["map"]),
        "map_asset": str(route_metadata["map"]),
        "trial_count": int(trials["count"]),
    }
    return dict(config), resolved


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def duration_spread_gate(
    durations_s: Sequence[float], maximum_fraction_of_median: float
) -> dict[str, Any]:
    if len(durations_s) != 3 or any(float(value) <= 0.0 for value in durations_s):
        return {
            "pass": False,
            "durations_s": [float(value) for value in durations_s],
            "median_s": None,
            "spread_s": None,
            "spread_fraction_of_median": None,
        }
    values = [float(value) for value in durations_s]
    median = statistics.median(values)
    spread = max(values) - min(values)
    fraction = spread / median
    return {
        "pass": fraction <= float(maximum_fraction_of_median),
        "durations_s": values,
        "median_s": median,
        "spread_s": spread,
        "spread_fraction_of_median": fraction,
    }


def _event(
    events: list[dict[str, Any]],
    *,
    experiment_id: str,
    trial_id: str,
    event_type: str,
    frame_id: int | None,
    sim_time_s: float | None,
    route_index: int | None,
    unwrapped_progress_m: float,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    events.append(
        {
            "experiment_id": experiment_id,
            "trial_id": trial_id,
            "event_id": len(events) + 1,
            "event_type": event_type,
            "frame_id": "" if frame_id is None else int(frame_id),
            "sim_time_s": "" if sim_time_s is None else float(sim_time_s),
            "route_index": "" if route_index is None else int(route_index),
            "unwrapped_progress_m": float(unwrapped_progress_m),
            "status": status,
            "details": json.dumps(details or {}, sort_keys=True, allow_nan=False),
        }
    )


def _speed_mps(velocity: Any) -> float:
    return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))


def _chase_spectator(carla: Any, world: Any, actor: Any, config: Mapping[str, Any]) -> None:
    if not bool(config["enabled"]):
        return
    followed = actor.get_transform()
    forward = followed.get_forward_vector()
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(
                x=float(followed.location.x) - float(config["behind_m"]) * float(forward.x),
                y=float(followed.location.y) - float(config["behind_m"]) * float(forward.y),
                z=float(followed.location.z) + float(config["above_m"]),
            ),
            carla.Rotation(
                pitch=float(config["pitch_deg"]),
                yaw=float(followed.rotation.yaw),
                roll=0.0,
            ),
        )
    )


def _spawn_pose_audit(transform: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    position_error = math.sqrt(
        (float(transform.location.x) - float(expected["x_m"])) ** 2
        + (float(transform.location.y) - float(expected["y_m"])) ** 2
        + (float(transform.location.z) - float(expected["z_m"])) ** 2
    )
    heading_error = abs(
        wrap_degrees(float(transform.rotation.yaw) - float(expected["yaw_deg"]))
    )
    return {
        "pass": bool(
            position_error <= float(expected["position_tolerance_m"])
            and heading_error <= float(expected["heading_tolerance_deg"])
        ),
        "position_error_m": position_error,
        "heading_error_deg": heading_error,
        "observed": {
            "x_m": float(transform.location.x),
            "y_m": float(transform.location.y),
            "z_m": float(transform.location.z),
            "yaw_deg": float(transform.rotation.yaw),
        },
    }


def _preexisting_dynamic_actors(world: Any) -> list[dict[str, Any]]:
    actors: dict[int, Any] = {}
    for pattern in ("vehicle.*", "walker.pedestrian.*", "sensor.*"):
        for actor in world.get_actors().filter(pattern):
            actors[int(actor.id)] = actor
    return [
        {
            "actor_id": actor_id,
            "type_id": str(actor.type_id),
            "role_name": str(actor.attributes.get("role_name", "")),
        }
        for actor_id, actor in sorted(actors.items())
    ]


def _destroy_actor(actor: Any) -> None:
    if actor is None:
        return
    try:
        if str(getattr(actor, "type_id", "")).startswith("sensor."):
            actor.stop()
    except (AttributeError, RuntimeError):
        pass
    try:
        actor.destroy()
    except RuntimeError:
        pass


def _cleanup_audit(
    world: Any,
    actor_ids: Sequence[int],
    *,
    tick_timeout_s: float,
    fixed_delta_s: float,
    maximum_sim_s: float,
) -> dict[str, Any]:
    """Wait bounded simulated time for destroyed owned actors to disappear."""

    tick_error = None
    cleanup_ticks = 0

    def remaining_live_actors() -> list[dict[str, Any]]:
        remaining: list[dict[str, Any]] = []
        for actor_id in actor_ids:
            try:
                actor = world.get_actor(int(actor_id))
            except RuntimeError:
                actor = None
            if actor is None:
                continue
            try:
                is_alive = bool(actor.is_alive)
            except (AttributeError, RuntimeError):
                is_alive = True
            if is_alive:
                remaining.append(
                    {
                        "actor_id": int(actor.id),
                        "type_id": str(actor.type_id),
                        "is_alive": True,
                    }
                )
        return remaining

    remaining = remaining_live_actors()
    maximum_ticks = int(math.ceil(float(maximum_sim_s) / float(fixed_delta_s)))
    while remaining and cleanup_ticks < maximum_ticks:
        try:
            world.tick(float(tick_timeout_s))
        except RuntimeError as exc:
            tick_error = f"{type(exc).__name__}: {exc}"
            break
        cleanup_ticks += 1
        remaining = remaining_live_actors()
    return {
        "pass": not remaining and tick_error is None,
        "owned_actor_ids": [int(value) for value in actor_ids],
        "remaining_owned_actors": remaining,
        "cleanup_tick_error": tick_error,
        "cleanup_ticks": cleanup_ticks,
        "cleanup_elapsed_sim_s": cleanup_ticks * float(fixed_delta_s),
        "cleanup_max_sim_s": float(maximum_sim_s),
    }


def trial_machine_gates(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    acceptance = config["acceptance"]
    lap = config["route"]["lap_detector"]
    progress_ratio = summary.get("unwrapped_progress_ratio")
    cadence = summary.get("observed_sim_control_rate_hz")
    delta_error = summary.get("maximum_sim_delta_error_s")
    p95 = summary.get("cross_track_p95_m")
    maximum_xtrack = summary.get("maximum_cross_track_m")
    return {
        "spawn_pose_exact": bool(summary.get("spawn_pose_audit", {}).get("pass")),
        "exactly_one_completed_lap": int(summary.get("lap_count", -1)) == 1,
        "exactly_one_wrap_after_arming": bool(
            int(summary.get("wrap_count", -1)) == int(lap["required_wrap_count"])
            and int(summary.get("wrap_after_arming_count", -1))
            == int(lap["required_wrap_count"])
        ),
        "ordered_route_sequence": int(summary.get("sequence_error_count", -1)) == 0,
        "unwrapped_progress_95_to_110_percent": bool(
            progress_ratio is not None
            and float(acceptance["minimum_unwrapped_progress_ratio"])
            <= float(progress_ratio)
            <= float(acceptance["maximum_unwrapped_progress_ratio"])
        ),
        "zero_false_completion": int(summary.get("false_completion_count", -1)) == 0,
        "return_position": bool(
            summary.get("return_position_error_m") is not None
            and float(summary["return_position_error_m"]) <= float(lap["completion_radius_m"])
        ),
        "return_heading": bool(
            summary.get("return_heading_error_deg") is not None
            and float(summary["return_heading_error_deg"])
            <= float(lap["completion_heading_tolerance_deg"])
        ),
        "cross_track_p95": bool(
            p95 is not None and float(p95) <= float(acceptance["maximum_cross_track_p95_m"])
        ),
        "no_persistent_divergence": bool(
            float(summary.get("maximum_continuous_divergence_s", math.inf))
            < float(acceptance["maximum_continuous_divergence_s"])
        ),
        "absolute_cross_track": bool(
            maximum_xtrack is not None
            and float(maximum_xtrack) <= float(acceptance["maximum_cross_track_error_m"])
        ),
        "zero_collisions": int(summary.get("collision_count", -1)) == 0,
        "final_collision_flush": bool(
            summary.get("collision_flush", {}).get("pass")
            and int(summary.get("collision_flush", {}).get("flush_tick_count", -1))
            == int(config["clock"]["collision_flush_ticks"])
        ),
        "no_unexplained_stall": bool(
            float(summary.get("maximum_continuous_stall_s", math.inf))
            <= float(acceptance["maximum_continuous_stall_s"])
        ),
        "duration_below_limit": bool(
            summary.get("duration_s") is not None
            and float(summary["duration_s"]) < float(config["trials"]["maximum_duration_s"])
        ),
        "control_cadence": bool(
            cadence is not None
            and abs(float(cadence) - float(config["clock"]["control_hz"]))
            <= float(acceptance["maximum_control_rate_error_hz"])
            and delta_error is not None
            and float(delta_error) <= float(acceptance["maximum_sim_delta_error_s"])
        ),
        "monotonic_unique_world_frames": int(summary.get("nonmonotonic_frame_count", -1)) == 0,
        "no_pose_jump": bool(
            float(summary.get("maximum_tick_displacement_m", math.inf))
            <= float(acceptance["maximum_tick_displacement_m"])
        ),
        "owned_actor_cleanup": bool(summary.get("cleanup", {}).get("pass")),
        "no_runtime_error": summary.get("runtime_error") is None,
    }


def _run_trial(
    *,
    carla: Any,
    world: Any,
    config: Mapping[str, Any],
    points: Sequence[RoutePoint],
    route_length_m: float,
    trial_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    experiment_id = str(config["experiment_id"])
    trace_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    mailbox = CollisionMailbox()
    actors: list[Any] = []
    actor_ids: list[int] = []
    ego: Any = None
    collision_sensor: Any = None
    pacer: MonotonicTickPacer | None = None
    runtime_error: str | None = None
    runtime_traceback: str | None = None
    spawn_audit: dict[str, Any] | None = None
    cleanup: dict[str, Any] = {
        "pass": False,
        "owned_actor_ids": [],
        "remaining_owned_actors": [],
        "cleanup_tick_error": "cleanup_not_run",
    }
    detector: ArmedLapDetector | None = None
    progress: OrderedRouteProgress | None = None
    last_lap_state: dict[str, Any] | None = None
    current_stall_s = 0.0
    current_stall_ticks = 0
    maximum_stall_s = 0.0
    current_divergence_s = 0.0
    current_divergence_ticks = 0
    maximum_divergence_s = 0.0
    maximum_tick_displacement_m = 0.0
    previous_xy: tuple[float, float] | None = None
    previous_frame: int | None = None
    nonmonotonic_frame_count = 0
    stop_reason: str | None = None
    seen_collisions = 0
    collision_flush: dict[str, Any] = {
        "pass": False,
        "flush_tick_count": 0,
        "flush_frame_id": None,
        "mailbox_count_before": 0,
        "mailbox_count_after": 0,
        "new_collision_rows": [],
        "error": "flush_not_run",
    }

    try:
        route = config["route"]
        clock = config["clock"]
        acceptance = config["acceptance"]
        spawn_points = world.get_map().get_spawn_points()
        spawn_index = int(route["spawn_index"])
        if not 0 <= spawn_index < len(spawn_points):
            raise RouteQualificationError("frozen spawn index is unavailable")
        spawn_transform = spawn_points[spawn_index]
        spawn_audit = _spawn_pose_audit(spawn_transform, route["expected_spawn"])
        if not spawn_audit["pass"]:
            raise RouteQualificationError(f"spawn-pose drift: {spawn_audit}")

        blueprint = select_exact_blueprint(
            world.get_blueprint_library(), str(route["ego_blueprint"])
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", str(route["ego_role_name"]))
        ego = world.try_spawn_actor(blueprint, spawn_transform)
        if ego is None:
            raise RouteQualificationError("ego spawn failed at index 55")
        actors.append(ego)
        actor_ids.append(int(ego.id))
        ego.set_autopilot(False)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(mailbox.callback)
        actors.append(collision_sensor)
        actor_ids.append(int(collision_sensor.id))

        pacer = MonotonicTickPacer(float(clock["real_time_tick_period_s"]))
        for _unused in range(int(config["trials"]["warmup_ticks"])):
            _chase_spectator(carla, world, ego, config["spectator"])
            pacer.wait()
            world.tick(float(clock["world_tick_timeout_s"]))

        transform = ego.get_transform()
        controller = ClosedDirectRouteController(points, route["controller"])
        controller.reset(transform.location.x, transform.location.y)
        assert controller.index is not None
        progress = OrderedRouteProgress(points, controller.index)
        detector = ArmedLapDetector(route_length_m, route["lap_detector"])
        last_lap_state = detector.update(
            x_m=float(transform.location.x),
            y_m=float(transform.location.y),
            yaw_deg=float(transform.rotation.yaw),
            unwrapped_progress_m=0.0,
            wrap_count=0,
            wrap_after_arming_count=0,
        )
        start_snapshot = world.get_snapshot()
        start_time = float(start_snapshot.timestamp.elapsed_seconds)
        last_sim_time = start_time
        previous_frame = int(start_snapshot.frame)
        previous_xy = (float(transform.location.x), float(transform.location.y))
        _event(
            event_rows,
            experiment_id=experiment_id,
            trial_id=trial_id,
            event_type="TRIAL_STARTED",
            frame_id=previous_frame,
            sim_time_s=start_time,
            route_index=controller.index,
            unwrapped_progress_m=0.0,
            status="INFO",
        )
        maximum_ticks = int(
            math.ceil(float(config["trials"]["maximum_duration_s"]) * float(clock["control_hz"]))
        )

        for tick_index in range(1, maximum_ticks + 1):
            transform = ego.get_transform()
            command = controller.command(
                x_m=float(transform.location.x),
                y_m=float(transform.location.y),
                yaw_deg=float(transform.rotation.yaw),
                speed_mps=_speed_mps(ego.get_velocity()),
                target_speed_mps=float(route["target_speed_mps"]),
            )
            for transition in command.transitions:
                transition_state = progress.observe(
                    transition, armed_before=bool(detector.armed)
                )
                if not transition_state["ordered"]:
                    _event(
                        event_rows,
                        experiment_id=experiment_id,
                        trial_id=trial_id,
                        event_type="ORDERED_PROGRESS_ERROR",
                        frame_id=previous_frame,
                        sim_time_s=last_sim_time,
                        route_index=command.route_index,
                        unwrapped_progress_m=progress.unwrapped_progress_m,
                        status="FAIL",
                        details={"transition": list(transition)},
                    )
                if transition_state["wrapped"]:
                    _event(
                        event_rows,
                        experiment_id=experiment_id,
                        trial_id=trial_id,
                        event_type="ROUTE_WRAP",
                        frame_id=previous_frame,
                        sim_time_s=last_sim_time,
                        route_index=command.route_index,
                        unwrapped_progress_m=progress.unwrapped_progress_m,
                        status=("INFO" if transition_state["armed_before_wrap"] else "FAIL"),
                        details={
                            "wrap_count": progress.wrap_count,
                            "armed_before_wrap": transition_state["armed_before_wrap"],
                        },
                    )

            ego.apply_control(
                carla.VehicleControl(
                    throttle=command.throttle,
                    steer=command.steer,
                    brake=command.brake,
                    hand_brake=False,
                )
            )
            _chase_spectator(carla, world, ego, config["spectator"])
            pacing = pacer.wait()
            wall_start = time.monotonic()
            frame_id = int(world.tick(float(clock["world_tick_timeout_s"])))
            tick_wall_s = time.monotonic() - wall_start
            snapshot = world.get_snapshot()
            sim_time = float(snapshot.timestamp.elapsed_seconds)
            sim_delta = sim_time - last_sim_time
            elapsed_s = sim_time - start_time
            if frame_id <= int(previous_frame):
                nonmonotonic_frame_count += 1
                stop_reason = "NONMONOTONIC_WORLD_FRAME"
            previous_frame = frame_id
            last_sim_time = sim_time

            transform = ego.get_transform()
            speed = _speed_mps(ego.get_velocity())
            current_xy = (float(transform.location.x), float(transform.location.y))
            tick_displacement = math.hypot(
                current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1]
            )
            previous_xy = current_xy
            maximum_tick_displacement_m = max(maximum_tick_displacement_m, tick_displacement)
            projection = project_to_closed_route(current_xy[0], current_xy[1], points)

            was_exited = detector.exited_start_gate
            was_armed = detector.armed
            was_completed = detector.completed
            last_lap_state = detector.update(
                x_m=current_xy[0],
                y_m=current_xy[1],
                yaw_deg=float(transform.rotation.yaw),
                unwrapped_progress_m=progress.unwrapped_progress_m,
                wrap_count=progress.wrap_count,
                wrap_after_arming_count=progress.wrap_after_arming_count,
            )
            if not was_exited and detector.exited_start_gate:
                _event(
                    event_rows,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    event_type="START_GATE_EXITED",
                    frame_id=frame_id,
                    sim_time_s=sim_time,
                    route_index=command.route_index,
                    unwrapped_progress_m=progress.unwrapped_progress_m,
                    status="INFO",
                )
            if not was_armed and detector.armed:
                _event(
                    event_rows,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    event_type="LAP_ARMED",
                    frame_id=frame_id,
                    sim_time_s=sim_time,
                    route_index=command.route_index,
                    unwrapped_progress_m=progress.unwrapped_progress_m,
                    status="INFO",
                )
            if not was_completed and detector.completed:
                _event(
                    event_rows,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    event_type="LAP_COMPLETED",
                    frame_id=frame_id,
                    sim_time_s=sim_time,
                    route_index=command.route_index,
                    unwrapped_progress_m=progress.unwrapped_progress_m,
                    status="INFO",
                )

            if elapsed_s >= float(acceptance["stall_grace_s"]) and speed < float(
                acceptance["stall_speed_threshold_mps"]
            ):
                current_stall_ticks += 1
            else:
                current_stall_ticks = 0
            current_stall_s = current_stall_ticks * float(clock["fixed_delta_seconds"])
            maximum_stall_s = max(maximum_stall_s, current_stall_s)
            if projection.cross_track_m > float(
                acceptance["divergence_cross_track_threshold_m"]
            ):
                current_divergence_ticks += 1
            else:
                current_divergence_ticks = 0
            current_divergence_s = current_divergence_ticks * float(
                clock["fixed_delta_seconds"]
            )
            maximum_divergence_s = max(maximum_divergence_s, current_divergence_s)

            new_collisions = mailbox.rows()[seen_collisions:]
            for collision in new_collisions:
                _event(
                    event_rows,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    event_type="COLLISION",
                    frame_id=int(collision["frame_id"]),
                    sim_time_s=sim_time,
                    route_index=command.route_index,
                    unwrapped_progress_m=progress.unwrapped_progress_m,
                    status="FAIL",
                    details=collision,
                )
            seen_collisions += len(new_collisions)

            trace_rows.append(
                {
                    "experiment_id": experiment_id,
                    "trial_id": trial_id,
                    "frame_id": frame_id,
                    "sim_time_s": sim_time,
                    "elapsed_s": elapsed_s,
                    "sim_delta_s": sim_delta,
                    "tick_wall_s": tick_wall_s,
                    "pacing_sleep_s": pacing["sleep_s"],
                    "pacing_lateness_s": pacing["lateness_s"],
                    "future_decision_slot": int(tick_index % 2 == 0),
                    "ego_x": current_xy[0],
                    "ego_y": current_xy[1],
                    "ego_z": float(transform.location.z),
                    "ego_yaw_deg": float(transform.rotation.yaw),
                    "ego_speed_mps": speed,
                    "route_index": command.route_index,
                    "route_target_index": command.target_index,
                    "unwrapped_progress_m": progress.unwrapped_progress_m,
                    "lap_count": detector.lap_count,
                    "wrap_count": progress.wrap_count,
                    "cross_track_m": projection.cross_track_m,
                    "heading_error_deg": math.degrees(command.heading_error_rad),
                    "return_position_error_m": last_lap_state["start_gate_distance_m"],
                    "return_heading_error_deg": last_lap_state["return_heading_error_deg"],
                    "throttle": command.throttle,
                    "steer": command.steer,
                    "brake": command.brake,
                    "collision_count": mailbox.count(),
                    "stall_s": current_stall_s,
                    "divergence_s": current_divergence_s,
                    "lap_armed": int(detector.armed),
                }
            )

            if mailbox.count() > 0:
                stop_reason = "COLLISION"
            elif projection.cross_track_m > float(acceptance["maximum_cross_track_error_m"]):
                stop_reason = "ABSOLUTE_CROSS_TRACK_BREACH"
            elif current_divergence_s >= float(
                acceptance["maximum_continuous_divergence_s"]
            ):
                stop_reason = "PERSISTENT_DIVERGENCE"
            elif current_stall_s > float(acceptance["maximum_continuous_stall_s"]):
                stop_reason = "PERSISTENT_STALL"
            elif tick_displacement > float(acceptance["maximum_tick_displacement_m"]):
                stop_reason = "POSE_JUMP"
            elif detector.false_completion_count > 0:
                stop_reason = "FALSE_COMPLETION_ATTEMPT"
            elif progress.sequence_error_count > 0:
                stop_reason = "ORDERED_PROGRESS_ERROR"
            elif progress.wrap_count > int(route["lap_detector"]["required_wrap_count"]):
                stop_reason = "EXCESS_ROUTE_WRAP"
            elif detector.completed:
                stop_reason = "LAP_COMPLETED"
            if stop_reason is not None:
                _event(
                    event_rows,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    event_type="TRIAL_STOP",
                    frame_id=frame_id,
                    sim_time_s=sim_time,
                    route_index=command.route_index,
                    unwrapped_progress_m=progress.unwrapped_progress_m,
                    status=("INFO" if stop_reason == "LAP_COMPLETED" else "FAIL"),
                    details={"reason": stop_reason},
                )
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                break
        else:
            stop_reason = "DURATION_TIMEOUT"
            _event(
                event_rows,
                experiment_id=experiment_id,
                trial_id=trial_id,
                event_type="TRIAL_STOP",
                frame_id=previous_frame,
                sim_time_s=last_sim_time,
                route_index=controller.index,
                unwrapped_progress_m=progress.unwrapped_progress_m,
                status="FAIL",
                details={"reason": stop_reason},
            )
    except (Exception, KeyboardInterrupt) as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        runtime_traceback = traceback.format_exc()
        _event(
            event_rows,
            experiment_id=experiment_id,
            trial_id=trial_id,
            event_type="RUNTIME_ERROR",
            frame_id=previous_frame,
            sim_time_s=None,
            route_index=(progress.expected_index if progress is not None else None),
            unwrapped_progress_m=(progress.unwrapped_progress_m if progress else 0.0),
            status="FAIL",
            details={"error": runtime_error},
        )
    finally:
        if ego is not None and collision_sensor is not None:
            try:
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if pacer is not None:
                    pacer.wait()
                collision_flush = flush_collision_tick(
                    world,
                    mailbox,
                    tick_timeout_s=float(config["clock"]["world_tick_timeout_s"]),
                    settle_s=float(config["clock"]["collision_flush_settle_s"]),
                )
                final_unseen_collisions = mailbox.rows()[seen_collisions:]
                collision_flush["drained_unseen_collision_count"] = len(
                    final_unseen_collisions
                )
                _event(
                    event_rows,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    event_type="COLLISION_FLUSH",
                    frame_id=collision_flush["flush_frame_id"],
                    sim_time_s=None,
                    route_index=(
                        progress.expected_index if progress is not None else None
                    ),
                    unwrapped_progress_m=(
                        progress.unwrapped_progress_m if progress else 0.0
                    ),
                    status=("INFO" if collision_flush["pass"] else "FAIL"),
                    details={
                        "flush_tick_count": collision_flush["flush_tick_count"],
                        "mailbox_count_before": collision_flush[
                            "mailbox_count_before"
                        ],
                        "mailbox_count_after": collision_flush[
                            "mailbox_count_after"
                        ],
                        "drained_unseen_collision_count": collision_flush[
                            "drained_unseen_collision_count"
                        ],
                    },
                )
                for collision in final_unseen_collisions:
                    _event(
                        event_rows,
                        experiment_id=experiment_id,
                        trial_id=trial_id,
                        event_type="COLLISION_FINAL_FLUSH",
                        frame_id=int(collision["frame_id"]),
                        sim_time_s=None,
                        route_index=(
                            progress.expected_index if progress is not None else None
                        ),
                        unwrapped_progress_m=(
                            progress.unwrapped_progress_m if progress else 0.0
                        ),
                        status="FAIL",
                        details=collision,
                    )
                if final_unseen_collisions:
                    stop_reason = "COLLISION_FINAL_FLUSH"
                if not collision_flush["pass"] and runtime_error is None:
                    runtime_error = (
                        "CollisionFlushError: " + str(collision_flush["error"])
                    )
            except Exception as exc:
                collision_flush["error"] = f"{type(exc).__name__}: {exc}"
                if runtime_error is None:
                    runtime_error = f"CollisionFlushError: {exc}"
        for actor in reversed(actors):
            _destroy_actor(actor)
        cleanup = _cleanup_audit(
            world,
            actor_ids,
            tick_timeout_s=float(config["clock"]["world_tick_timeout_s"]),
            fixed_delta_s=float(config["clock"]["fixed_delta_seconds"]),
            maximum_sim_s=float(config["acceptance"]["cleanup_max_sim_s"]),
        )
        _event(
            event_rows,
            experiment_id=experiment_id,
            trial_id=trial_id,
            event_type="CLEANUP",
            frame_id=previous_frame,
            sim_time_s=None,
            route_index=(progress.expected_index if progress is not None else None),
            unwrapped_progress_m=(progress.unwrapped_progress_m if progress else 0.0),
            status=("INFO" if cleanup["pass"] else "FAIL"),
            details=cleanup,
        )

    sim_deltas = [float(row["sim_delta_s"]) for row in trace_rows]
    xtracks = [float(row["cross_track_m"]) for row in trace_rows]
    duration_s = float(trace_rows[-1]["elapsed_s"]) if trace_rows else None
    observed_rate = (
        (len(trace_rows) - 1)
        / (float(trace_rows[-1]["sim_time_s"]) - float(trace_rows[0]["sim_time_s"]))
        if len(trace_rows) >= 2
        and float(trace_rows[-1]["sim_time_s"]) > float(trace_rows[0]["sim_time_s"])
        else None
    )
    unwrapped = progress.unwrapped_progress_m if progress is not None else 0.0
    summary: dict[str, Any] = {
        "trial_id": trial_id,
        "machine_status": "PENDING",
        "spawn_pose_audit": spawn_audit,
        "stop_reason": stop_reason,
        "route_length_m": route_length_m,
        "duration_s": duration_s,
        "tick_count": len(trace_rows),
        "lap_count": detector.lap_count if detector is not None else 0,
        "wrap_count": progress.wrap_count if progress is not None else 0,
        "wrap_after_arming_count": (
            progress.wrap_after_arming_count if progress is not None else 0
        ),
        "sequence_error_count": (
            progress.sequence_error_count if progress is not None else 0
        ),
        "false_completion_count": (
            detector.false_completion_count if detector is not None else 0
        ),
        "unwrapped_progress_m": unwrapped,
        "unwrapped_progress_ratio": unwrapped / route_length_m,
        "return_position_error_m": (
            float(last_lap_state["start_gate_distance_m"])
            if last_lap_state is not None and trace_rows
            else None
        ),
        "return_heading_error_deg": (
            float(last_lap_state["return_heading_error_deg"])
            if last_lap_state is not None and trace_rows
            else None
        ),
        "cross_track_p95_m": _percentile(xtracks, 0.95),
        "maximum_cross_track_m": max(xtracks) if xtracks else None,
        "maximum_continuous_divergence_s": maximum_divergence_s,
        "maximum_continuous_stall_s": maximum_stall_s,
        "maximum_tick_displacement_m": maximum_tick_displacement_m,
        "collision_count": mailbox.count(),
        "collision_flush": collision_flush,
        "nonmonotonic_frame_count": nonmonotonic_frame_count,
        "configured_control_rate_hz": float(config["clock"]["control_hz"]),
        "observed_sim_control_rate_hz": observed_rate,
        "maximum_sim_delta_error_s": (
            max(
                abs(value - float(config["clock"]["fixed_delta_seconds"]))
                for value in sim_deltas
            )
            if sim_deltas
            else None
        ),
        "wall_tick_p50_s": _percentile(
            [float(row["tick_wall_s"]) for row in trace_rows], 0.50
        ),
        "wall_tick_p95_s": _percentile(
            [float(row["tick_wall_s"]) for row in trace_rows], 0.95
        ),
        "cleanup": cleanup,
        "runtime_error": runtime_error,
        "runtime_traceback": runtime_traceback,
    }
    gates = trial_machine_gates(summary, config)
    summary["machine_gates"] = gates
    summary["machine_status"] = (
        "ACCEPTED_PENDING_MANUAL_REVIEW" if all(gates.values()) else "FAILED"
    )
    return trace_rows, event_rows, summary


def manual_review_template(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANUAL_REVIEW_SCHEMA,
        "reviewer": str(config["manual_review"]["required_reviewer"]),
        "reviewed_at": None,
        "overall_verdict": "PENDING",
        "trials": {
            f"trial_{index:02d}": {
                "verdict": "PENDING",
                "checks": {
                    str(name): None
                    for name in config["manual_review"]["required_checks"]
                },
                "anomalies": [],
                "notes": "",
            }
            for index in range(1, int(config["trials"]["count"]) + 1)
        },
        "instructions": (
            "Watch all three trials. Copy this file to manual_review.json; fill "
            "reviewed_at, eight booleans and PASS/FAIL per trial, anomalies with "
            "trial/time/route region, and the consistent overall verdict."
        ),
    }


def _fallback_manual_template(error: str) -> dict[str, Any]:
    return {
        "schema": MANUAL_REVIEW_SCHEMA,
        "reviewer": "Abiodun",
        "reviewed_at": None,
        "overall_verdict": "NOT_APPLICABLE_MACHINE_FAILURE",
        "trials": {},
        "instructions": "Manual review is unavailable because preflight failed.",
        "machine_error": str(error),
    }


def _create_run_dir(output_root: Path) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for _unused in range(10):
        run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            time.sleep(0.001)
    raise RouteQualificationError("could not allocate unique run directory")


def _map_matches(observed: str, required: str) -> bool:
    return str(observed).rstrip("/").split("/")[-1] == str(required).split("/")[-1]


def _publish_bundle(
    *,
    run_dir: Path,
    raw_config_text: str,
    manifest: Mapping[str, Any],
    route_contract: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    machine_review: Mapping[str, Any],
    manual_template: Mapping[str, Any],
) -> None:
    status = str(machine_review["status"])
    if status not in {"REVIEW_REQUIRED", "FAILED"}:
        raise RouteQualificationError(f"invalid terminal status: {status}")
    atomic_write_new_text(run_dir / "resolved_config.yaml", raw_config_text)
    atomic_write_new_json(run_dir / "manifest.json", manifest)
    atomic_write_new_json(run_dir / "route_contract.json", route_contract)
    atomic_write_new_text(
        run_dir / "route_trace.csv", rows_to_csv_text(TRACE_FIELDS, trace_rows)
    )
    atomic_write_new_text(
        run_dir / "route_events.csv", rows_to_csv_text(EVENT_FIELDS, event_rows)
    )
    atomic_write_new_json(run_dir / "ROUTE_MACHINE_REVIEW.json", machine_review)
    atomic_write_new_json(run_dir / "manual_review_template.json", manual_template)
    atomic_write_new_json(
        run_dir / f"{status}.json",
        {
            "schema": "scenesense.ue_route_terminal.v1",
            "created_at": utc_now(),
            "status": status,
            "machine_review": "ROUTE_MACHINE_REVIEW.json",
        },
    )


def run_qualification(config_path: Path, output_root: Path) -> tuple[Path, str]:
    """Run three trials or preserve an immutable FAILED evidence bundle."""

    run_dir = _create_run_dir(output_root)
    config_path = Path(config_path).expanduser().resolve()
    raw_config_error: str | None = None
    try:
        raw_config_text = config_path.read_text(encoding="utf-8")
    except Exception as exc:
        raw_config_error = f"{type(exc).__name__}: {exc}"
        raw_config_text = json_text(
            {"config_path": str(config_path), "config_read_error": raw_config_error}
        )
    created_at = utc_now()
    trace_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    trial_summaries: list[dict[str, Any]] = []
    config: dict[str, Any] | None = None
    resolved: dict[str, Any] | None = None
    top_error: str | None = None
    top_traceback: str | None = None
    world: Any = None
    original_settings: Any = None

    try:
        if raw_config_error is not None:
            raise RouteQualificationError(raw_config_error)
        config, resolved = load_and_validate_config(config_path)
        try:
            import carla  # type: ignore
        except ImportError as exc:
            raise RouteQualificationError(
                "CARLA PythonAPI unavailable; use the CARLA environment interpreter"
            ) from exc
        client = carla.Client(str(config["carla"]["host"]), int(config["carla"]["port"]))
        client.set_timeout(float(config["carla"]["timeout_s"]))
        world = (
            client.load_world(str(config["carla"]["map"]), True)
            if bool(config["carla"]["reload_world_at_start"])
            else client.get_world()
        )
        if not _map_matches(str(world.get_map().name), str(config["carla"]["map"])):
            raise RouteQualificationError("loaded CARLA map violates frozen contract")
        if bool(config["scene"]["forbid_preexisting_vehicles_walkers_and_sensors"]):
            preexisting = _preexisting_dynamic_actors(world)
            if preexisting:
                raise RouteQualificationError(
                    f"ego-only preflight found dynamic actors: {preexisting}"
                )
        # Resolve the exact frozen ID before printing a visible-trial start.
        # CARLA's BlueprintLibrary.find() raises only an opaque std::exception
        # for a missing ID in this build, so keep the actionable preflight here.
        select_exact_blueprint(
            world.get_blueprint_library(), str(config["route"]["ego_blueprint"])
        )
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(config["clock"]["fixed_delta_seconds"])
        world.apply_settings(settings)

        points = read_route_points(Path(resolved["progress_csv_path"]))
        for trial_index in range(1, int(config["trials"]["count"]) + 1):
            trial_id = f"trial_{trial_index:02d}"
            print(f"Starting visible route qualification {trial_id}/trial_03")
            trial_trace, trial_events, summary = _run_trial(
                carla=carla,
                world=world,
                config=config,
                points=points,
                route_length_m=float(resolved["closed_route_length_m"]),
                trial_id=trial_id,
            )
            trace_rows.extend(trial_trace)
            # Event IDs are per trial by contract; retain them when aggregating.
            event_rows.extend(trial_events)
            trial_summaries.append(summary)
            print(f"{trial_id} machine status: {summary['machine_status']}")
            if summary["machine_status"] == "FAILED":
                break
    except (Exception, KeyboardInterrupt) as exc:
        top_error = f"{type(exc).__name__}: {exc}"
        top_traceback = traceback.format_exc()
    finally:
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception as exc:
                restore_error = f"{type(exc).__name__}: {exc}"
                top_error = (
                    restore_error
                    if top_error is None
                    else f"{top_error}; world_restore={restore_error}"
                )

    if config is not None and resolved is not None:
        duration_gate = duration_spread_gate(
            [
                float(summary["duration_s"])
                for summary in trial_summaries
                if summary.get("duration_s") is not None
            ],
            float(config["acceptance"]["maximum_duration_spread_fraction_of_median"]),
        )
        run_gates = {
            "exactly_three_independent_trials": len(trial_summaries)
            == int(config["trials"]["count"]),
            "all_trial_machine_gates": bool(
                len(trial_summaries) == int(config["trials"]["count"])
                and all(
                    summary["machine_status"] == "ACCEPTED_PENDING_MANUAL_REVIEW"
                    for summary in trial_summaries
                )
            ),
            "duration_spread_within_five_percent": bool(duration_gate["pass"]),
            "no_orchestration_error": top_error is None,
        }
        status = "REVIEW_REQUIRED" if all(run_gates.values()) else "FAILED"
        manual_template = manual_review_template(config)
        route_contract = {
            "schema": ROUTE_CONTRACT_SCHEMA,
            "route_id": str(config["route"]["route_id"]),
            "resolved": resolved,
            "spawn_index": int(config["route"]["spawn_index"]),
            "target_speed_mps": float(config["route"]["target_speed_mps"]),
            "controller": config["route"]["controller"],
            "lap_detector": config["route"]["lap_detector"],
            "clock": config["clock"],
            "scene": config["scene"],
            "acceptance": config["acceptance"],
        }
    else:
        duration_gate = duration_spread_gate([], 0.05)
        run_gates = {
            "exactly_three_independent_trials": False,
            "all_trial_machine_gates": False,
            "duration_spread_within_five_percent": False,
            "no_orchestration_error": False,
        }
        status = "FAILED"
        manual_template = _fallback_manual_template(top_error or "config preflight failed")
        route_contract = {
            "schema": ROUTE_CONTRACT_SCHEMA,
            "status": "UNRESOLVED_CONFIG_FAILURE",
            "config_path": str(config_path),
        }

    if top_error is not None:
        _event(
            event_rows,
            experiment_id=(
                str(config["experiment_id"])
                if config is not None
                else "ue_route_qualification_v1"
            ),
            trial_id="orchestration",
            event_type="ORCHESTRATION_FAILURE",
            frame_id=None,
            sim_time_s=None,
            route_index=None,
            unwrapped_progress_m=0.0,
            status="FAIL",
            details={"error": top_error},
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": created_at,
        "experiment_id": (
            str(config["experiment_id"]) if config is not None else "ue_route_qualification_v1"
        ),
        "config_path": str(config_path),
        "resolved_contract": resolved,
        "artifact_contract": {
            "create_only": True,
            "machine_success_terminal": "REVIEW_REQUIRED.json",
            "machine_failure_terminal": "FAILED.json",
            "machine_pass_name_forbidden": True,
        },
    }
    machine_review = {
        "schema": MACHINE_REVIEW_SCHEMA,
        "created_at": utc_now(),
        "status": status,
        "trial_summaries": trial_summaries,
        "duration_spread_gate": duration_gate,
        "run_gates": run_gates,
        "orchestration_error": top_error,
        "orchestration_traceback": top_traceback,
        "manual_review_required_for_final_pass": status == "REVIEW_REQUIRED",
    }
    _publish_bundle(
        run_dir=run_dir,
        raw_config_text=raw_config_text,
        manifest=manifest,
        route_contract=route_contract,
        trace_rows=trace_rows,
        event_rows=event_rows,
        machine_review=machine_review,
        manual_template=manual_template,
    )
    return run_dir, status


def validate_manual_review(
    review: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if review.get("schema") != MANUAL_REVIEW_SCHEMA:
        raise RouteQualificationError("invalid manual review schema")
    if str(review.get("reviewer")) != str(config["manual_review"]["required_reviewer"]):
        raise RouteQualificationError("manual review has wrong reviewer")
    if not review.get("reviewed_at"):
        raise RouteQualificationError("manual review lacks reviewed_at")
    trials = _require_mapping(review.get("trials"), "manual_review.trials")
    required_checks = [str(value) for value in config["manual_review"]["required_checks"]]
    validated_trials: dict[str, Any] = {}
    for index in range(1, int(config["trials"]["count"]) + 1):
        trial_id = f"trial_{index:02d}"
        trial = _require_mapping(trials.get(trial_id), trial_id)
        checks = _require_mapping(trial.get("checks"), f"{trial_id}.checks")
        if set(checks) != set(required_checks) or any(
            not isinstance(checks[name], bool) for name in required_checks
        ):
            raise RouteQualificationError(
                f"{trial_id} must contain exactly eight boolean checks"
            )
        verdict = str(trial.get("verdict", "")).upper()
        derived_pass = all(bool(checks[name]) for name in required_checks)
        if verdict not in {"PASS", "FAIL"} or (verdict == "PASS") != derived_pass:
            raise RouteQualificationError(f"{trial_id} verdict contradicts its checks")
        validated_trials[trial_id] = {
            "verdict": verdict,
            "checks": dict(checks),
            "anomalies": list(trial.get("anomalies", [])),
            "notes": str(trial.get("notes", "")),
        }
    overall = str(review.get("overall_verdict", "")).upper()
    derived_overall = all(value["verdict"] == "PASS" for value in validated_trials.values())
    if overall not in {"PASS", "FAIL"} or (overall == "PASS") != derived_overall:
        raise RouteQualificationError("overall verdict contradicts trial verdicts")
    return {
        "reviewer": str(review["reviewer"]),
        "reviewed_at": str(review["reviewed_at"]),
        "overall_verdict": overall,
        "trials": validated_trials,
    }


def finalize_reviewed_run(run_dir: Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    if not (run_dir / "REVIEW_REQUIRED.json").is_file():
        raise RouteQualificationError("only REVIEW_REQUIRED evidence can be finalized")
    config = json.loads((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    review_path = run_dir / "manual_review.json"
    if not review_path.is_file():
        raise RouteQualificationError(f"manual review is missing: {review_path}")
    review = validate_manual_review(
        json.loads(review_path.read_text(encoding="utf-8")), config
    )
    passed = review["overall_verdict"] == "PASS"
    output = run_dir / ("ROUTE_QUALIFIED.json" if passed else "FAILED_MANUAL.json")
    atomic_write_new_json(
        output,
        {
            "schema": FINAL_REVIEW_SCHEMA,
            "created_at": utc_now(),
            "status": "PASS" if passed else "FAIL_MANUAL",
            "manual_review": review,
        },
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or finalize the ego-only UE route qualification."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the frozen contract without importing CARLA",
    )
    parser.add_argument("--finalize-reviewed-run", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.finalize_reviewed_run is not None:
            output = finalize_reviewed_run(args.finalize_reviewed_run)
            print(f"Route review decision: {output}")
            return 0
        if args.validate_only:
            _config, resolved = load_and_validate_config(args.config)
            print(json_text(resolved), end="")
            return 0
        run_dir, status = run_qualification(args.config, args.output_root)
        print(f"Route qualification evidence: {run_dir}")
        print(f"Machine status: {status}")
        return 0 if status == "REVIEW_REQUIRED" else 2
    except (OSError, ValueError, RouteQualificationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

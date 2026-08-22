#!/usr/bin/env python3
"""Offline, fail-closed contracts for the full-route AE64 perception pilot.

This module intentionally has no CARLA, torch, model-training, or evaluation
imports.  It validates externally supplied route evidence, builds dry-run
collection matrices, creates leakage-safe split manifests, and emits immutable
readiness records.  It cannot start collection, training, or test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TERMINAL_WAITING = "WAITING_FOR_CANONICAL_ROUTE"
SCHEMA_VERSION = "scenesense.perception_full_route_scaffold.v1"
FINAL_TEST_MARKER = "TEST_EVALUATION_AUTHORIZED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
SPLITS = ("train", "validation", "final_test")


class ContractError(ValueError):
    """A frozen contract is missing, inconsistent, or unsafe."""


class WaitingForCanonicalRoute(ContractError):
    """The canonical Route B artifact is absent or has not passed validation."""


class CreateOnlyError(FileExistsError):
    """A create-only path already exists."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_only_directory(path: Path | str) -> Path:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise CreateOnlyError(f"create-only directory already exists: {target}") from exc
    return target


def write_create_only(path: Path | str, payload: bytes) -> Path:
    target = Path(path)
    if not target.parent.is_dir():
        raise ContractError(f"parent directory does not exist: {target.parent}")
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise CreateOnlyError(f"create-only file already exists: {target}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def write_json_create_only(path: Path | str, value: Any) -> Path:
    return write_create_only(path, pretty_json_bytes(value))


def _load_document(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise WaitingForCanonicalRoute(f"canonical route file is missing: {path}") from exc
    if suffix == ".json":
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid route JSON: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ContractError("YAML route supplied but PyYAML is unavailable; supply canonical JSON or install PyYAML") from exc
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ContractError(f"invalid route YAML: {exc}") from exc
    else:
        raise ContractError("canonical route extension must be .json, .yaml, or .yml")
    if not isinstance(value, Mapping):
        raise ContractError("canonical route root must be an object")
    return value


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    return value


def _required(obj: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [name for name in fields if name not in obj or obj[name] is None]
    if missing:
        raise ContractError(f"{label} missing required fields: {', '.join(missing)}")


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value.strip()


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.sqrt(sum((float(a[axis]) - float(b[axis])) ** 2 for axis in ("x", "y", "z")))


def validate_route_document(route: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the semantic Route B contract without importing CARLA."""
    _required(
        route,
        (
            "schema",
            "route_id",
            "version",
            "map",
            "coordinate_frame",
            "route_direction",
            "waypoints",
            "segments",
            "ego_spawn_transform",
            "loop_closure",
            "route_length",
            "expected_duration",
            "qualification",
        ),
        "route",
    )
    if route["schema"] != "scenesense.canonical_route.v1":
        raise ContractError("route.schema must equal scenesense.canonical_route.v1")
    route_id = _nonempty_string(route["route_id"], "route.route_id")
    if "route_b" not in route_id.lower() and "route-b" not in route_id.lower():
        raise ContractError("route.route_id must identify canonical Route B")
    version = _nonempty_string(route["version"], "route.version")
    if not SEMVER_RE.fullmatch(version):
        raise ContractError("route.version must be semantic version text such as 1.0.0")

    map_spec = _object(route["map"], "route.map")
    _required(map_spec, ("town", "map_identifier"), "route.map")
    if map_spec["town"] != "Town10HD":
        raise ContractError("route.map.town must equal Town10HD")
    map_identifier = _nonempty_string(map_spec["map_identifier"], "route.map.map_identifier")
    if "town10hd" not in map_identifier.lower():
        raise ContractError("route.map.map_identifier must identify Town10HD")

    frame = _object(route["coordinate_frame"], "route.coordinate_frame")
    _required(frame, ("name", "units", "handedness", "axes"), "route.coordinate_frame")
    if frame["name"] != "carla_world" or frame["units"] != "meters" or frame["handedness"] != "left_handed":
        raise ContractError("coordinate frame must be CARLA world, meters, left_handed")
    axes = _object(frame["axes"], "route.coordinate_frame.axes")
    _required(axes, ("x", "y", "z"), "route.coordinate_frame.axes")
    if axes != {"x": "forward", "y": "right", "z": "up"}:
        raise ContractError("coordinate axes must explicitly be x=forward, y=right, z=up")

    direction = _object(route["route_direction"], "route.route_direction")
    _required(direction, ("direction", "waypoint_order"), "route.route_direction")
    if direction["direction"] not in {"forward", "reverse"}:
        raise ContractError("route.route_direction.direction must be forward or reverse")
    if direction["waypoint_order"] != "travel_order":
        raise ContractError("route.route_direction.waypoint_order must equal travel_order")

    waypoints = _array(route["waypoints"], "route.waypoints")
    if len(waypoints) < 2:
        raise ContractError("route.waypoints must contain at least two ordered waypoints")
    normalized_waypoints: List[Mapping[str, Any]] = []
    for index, raw in enumerate(waypoints):
        point = _object(raw, f"route.waypoints[{index}]")
        _required(point, ("sequence_index", "x", "y", "z", "route_segment_id"), f"route.waypoints[{index}]")
        if point["sequence_index"] != index:
            raise ContractError("waypoint sequence_index values must be contiguous and match travel order")
        for axis in ("x", "y", "z"):
            _number(point[axis], f"route.waypoints[{index}].{axis}")
        _nonempty_string(point["route_segment_id"], f"route.waypoints[{index}].route_segment_id")
        normalized_waypoints.append(point)

    segments = _array(route["segments"], "route.segments")
    if not segments:
        raise ContractError("route.segments must not be empty")
    segment_ids: List[str] = []
    previous_end = -1
    for index, raw in enumerate(segments):
        segment = _object(raw, f"route.segments[{index}]")
        _required(segment, ("segment_id", "start_waypoint_index", "end_waypoint_index"), f"route.segments[{index}]")
        segment_id = _nonempty_string(segment["segment_id"], f"route.segments[{index}].segment_id")
        if segment_id in segment_ids:
            raise ContractError(f"duplicate route segment_id: {segment_id}")
        start = segment["start_waypoint_index"]
        end = segment["end_waypoint_index"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end >= len(waypoints):
            raise ContractError(f"invalid waypoint range for segment {segment_id}")
        if index == 0 and start != 0:
            raise ContractError("first route segment must start at waypoint 0")
        if index > 0 and start != previous_end + 1:
            raise ContractError("route segment waypoint ranges must be contiguous and non-overlapping")
        previous_end = end
        segment_ids.append(segment_id)
    if previous_end != len(waypoints) - 1:
        raise ContractError("route segments must cover every waypoint")
    for index, point in enumerate(normalized_waypoints):
        segment_id = str(point["route_segment_id"])
        if segment_id not in segment_ids:
            raise ContractError(f"waypoint {index} references unknown segment {segment_id}")

    loop = _object(route["loop_closure"], "route.loop_closure")
    _required(loop, ("mode", "close_last_to_first", "position_tolerance_m", "heading_tolerance_deg"), "route.loop_closure")
    if loop["mode"] not in {"closed_loop", "open_route"}:
        raise ContractError("loop_closure.mode must be closed_loop or open_route")
    if not isinstance(loop["close_last_to_first"], bool):
        raise ContractError("loop_closure.close_last_to_first must be boolean")
    if (loop["mode"] == "closed_loop") != loop["close_last_to_first"]:
        raise ContractError("loop closure mode and close_last_to_first disagree")
    position_tolerance = _number(loop["position_tolerance_m"], "loop_closure.position_tolerance_m", minimum=0.0)
    _number(loop["heading_tolerance_deg"], "loop_closure.heading_tolerance_deg", minimum=0.0)
    seam_m = _distance(normalized_waypoints[-1], normalized_waypoints[0])
    if loop["mode"] == "closed_loop" and seam_m > position_tolerance:
        raise ContractError(f"closed-loop endpoint separation {seam_m:.6f} m exceeds tolerance {position_tolerance:.6f} m")

    spawn = _object(route["ego_spawn_transform"], "route.ego_spawn_transform")
    _required(spawn, ("location", "rotation", "route_start_waypoint_index", "route_start_position_tolerance_m"), "route.ego_spawn_transform")
    location = _object(spawn["location"], "route.ego_spawn_transform.location")
    rotation = _object(spawn["rotation"], "route.ego_spawn_transform.rotation")
    _required(location, ("x", "y", "z"), "route.ego_spawn_transform.location")
    _required(rotation, ("pitch", "yaw", "roll"), "route.ego_spawn_transform.rotation")
    for axis in ("x", "y", "z"):
        _number(location[axis], f"route.ego_spawn_transform.location.{axis}")
    for axis in ("pitch", "yaw", "roll"):
        _number(rotation[axis], f"route.ego_spawn_transform.rotation.{axis}")
    start_index = spawn["route_start_waypoint_index"]
    if not isinstance(start_index, int) or not 0 <= start_index < len(waypoints):
        raise ContractError("ego_spawn_transform.route_start_waypoint_index is invalid")
    spawn_tolerance = _number(spawn["route_start_position_tolerance_m"], "ego_spawn_transform.route_start_position_tolerance_m", minimum=0.0)
    spawn_distance = _distance(location, normalized_waypoints[start_index])
    if spawn_distance > spawn_tolerance:
        raise ContractError(f"ego spawn is {spawn_distance:.6f} m from route start, beyond {spawn_tolerance:.6f} m")

    length_spec = _object(route["route_length"], "route.route_length")
    _required(length_spec, ("declared_m", "calculation", "measurement_tolerance_m"), "route.route_length")
    declared_length = _number(length_spec["declared_m"], "route.route_length.declared_m", minimum=0.001)
    if length_spec["calculation"] != "polyline_waypoints_with_optional_loop_seam":
        raise ContractError("route_length.calculation must equal polyline_waypoints_with_optional_loop_seam")
    length_tolerance = _number(length_spec["measurement_tolerance_m"], "route.route_length.measurement_tolerance_m", minimum=0.0)
    computed_length = sum(_distance(normalized_waypoints[i - 1], normalized_waypoints[i]) for i in range(1, len(waypoints)))
    if loop["close_last_to_first"]:
        computed_length += seam_m
    if abs(declared_length - computed_length) > length_tolerance:
        raise ContractError(
            f"declared route length differs from waypoint polyline: declared={declared_length:.6f}, "
            f"computed={computed_length:.6f}, tolerance={length_tolerance:.6f}"
        )

    duration = _object(route["expected_duration"], "route.expected_duration")
    _required(duration, ("minimum_s", "nominal_s", "maximum_s"), "route.expected_duration")
    minimum_s = _number(duration["minimum_s"], "expected_duration.minimum_s", minimum=0.001)
    nominal_s = _number(duration["nominal_s"], "expected_duration.nominal_s", minimum=minimum_s)
    maximum_s = _number(duration["maximum_s"], "expected_duration.maximum_s", minimum=nominal_s)
    if not minimum_s <= nominal_s <= maximum_s:
        raise ContractError("expected duration must satisfy minimum <= nominal <= maximum")

    qualification = _object(route["qualification"], "route.qualification")
    _required(
        qualification,
        ("status", "qualified_by", "qualified_at_utc", "qualification_bundle_id", "qualification_manifest_sha256"),
        "route.qualification",
    )
    if qualification["status"] != "QUALIFIED":
        raise WaitingForCanonicalRoute("canonical Route B qualification.status must equal QUALIFIED")
    for field in ("qualified_by", "qualified_at_utc", "qualification_bundle_id"):
        _nonempty_string(qualification[field], f"route.qualification.{field}")
    manifest_sha = str(qualification["qualification_manifest_sha256"])
    if not SHA256_RE.fullmatch(manifest_sha):
        raise ContractError("qualification_manifest_sha256 must be a lowercase SHA-256 hex digest")

    return {
        "route_id": route_id,
        "route_version": version,
        "map_identifier": map_identifier,
        "waypoint_count": len(waypoints),
        "route_segment_count": len(segments),
        "computed_route_length_m": computed_length,
        "closed_loop": bool(loop["close_last_to_first"]),
        "qualification_manifest_sha256": manifest_sha,
    }


def verify_canonical_route(route_path: Path | str | None, expected_sha256: str | None) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Hash first, parse second, and fail closed before any CARLA startup."""
    if route_path is None:
        raise WaitingForCanonicalRoute("canonical Route B file is required")
    if expected_sha256 is None or not str(expected_sha256).strip():
        raise WaitingForCanonicalRoute("externally supplied canonical Route B SHA-256 is required")
    expected = str(expected_sha256).strip()
    if not SHA256_RE.fullmatch(expected):
        raise WaitingForCanonicalRoute("canonical Route B SHA-256 must be 64 lowercase hexadecimal characters")
    path = Path(route_path)
    if not path.is_file():
        raise WaitingForCanonicalRoute(f"canonical Route B file is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise WaitingForCanonicalRoute(f"canonical Route B hash drift: expected {expected}, observed {observed}")
    route = _load_document(path)
    summary = validate_route_document(route)
    summary.update({"route_file": str(path.resolve()), "route_file_sha256": observed, "status": "CANONICAL_ROUTE_VERIFIED"})
    return route, summary


def build_dry_run_matrix(collection_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand only density x seed-bundle; never UE action/network profiles."""
    profiles = _array(collection_config.get("density_profiles"), "collection.density_profiles")
    seed_bundles = _array(collection_config.get("seed_bundles"), "collection.seed_bundles")
    weather_cycle = _array(collection_config.get("weather_cycle"), "collection.weather_cycle")
    if not profiles or not seed_bundles or not weather_cycle:
        raise ContractError("density_profiles, seed_bundles, and weather_cycle must be non-empty")
    rows: List[Dict[str, Any]] = []
    seen_profiles: set[str] = set()
    seen_bundles: set[str] = set()
    for p_index, raw_profile in enumerate(profiles):
        profile = _object(raw_profile, f"density_profiles[{p_index}]")
        _required(profile, ("profile_id", "provisional_actor_counts"), f"density_profiles[{p_index}]")
        profile_id = _nonempty_string(profile["profile_id"], "profile_id")
        if profile_id not in {"low", "medium", "dense"} or profile_id in seen_profiles:
            raise ContractError("density profiles must be unique low, medium, and dense")
        seen_profiles.add(profile_id)
        counts = _object(profile["provisional_actor_counts"], f"density_profiles[{p_index}].provisional_actor_counts")
        _required(counts, ("vehicles", "pedestrians"), "provisional_actor_counts")
        if any(not isinstance(counts[key], int) or counts[key] < 0 for key in ("vehicles", "pedestrians")):
            raise ContractError("provisional actor counts must be non-negative integers")
        for s_index, raw_seed in enumerate(seed_bundles):
            seed = _object(raw_seed, f"seed_bundles[{s_index}]")
            _required(seed, ("seed_bundle_id", "carla_seed", "traffic_manager_seed", "reserved_split"), f"seed_bundles[{s_index}]")
            bundle_id = _nonempty_string(seed["seed_bundle_id"], "seed_bundle_id")
            if p_index == 0:
                if bundle_id in seen_bundles:
                    raise ContractError(f"duplicate seed bundle: {bundle_id}")
                seen_bundles.add(bundle_id)
            if seed["reserved_split"] not in SPLITS:
                raise ContractError(f"invalid reserved split for {bundle_id}")
            if not isinstance(seed["carla_seed"], int) or not isinstance(seed["traffic_manager_seed"], int):
                raise ContractError("CARLA and Traffic Manager seeds must be integers")
            weather = weather_cycle[(p_index * len(seed_bundles) + s_index) % len(weather_cycle)]
            row_key = f"{profile_id}|{bundle_id}|{seed['carla_seed']}|{seed['traffic_manager_seed']}|{weather}"
            rows.append(
                {
                    "matrix_index": len(rows),
                    "episode_plan_id": f"fullroute-v1-{profile_id}-{bundle_id}",
                    "density_profile": profile_id,
                    "provisional_actor_counts": dict(counts),
                    "seed_bundle_id": bundle_id,
                    "carla_seed": seed["carla_seed"],
                    "traffic_manager_seed": seed["traffic_manager_seed"],
                    "reserved_split": seed["reserved_split"],
                    "weather": weather,
                    "laps": collection_config.get("laps_per_episode"),
                    "route_id": None,
                    "route_sha256": None,
                    "scenario_id": f"fullroute:{profile_id}:{bundle_id}",
                    "determinism_key_sha256": sha256_bytes(row_key.encode("utf-8")),
                    "status": TERMINAL_WAITING,
                }
            )
    if seen_profiles != {"low", "medium", "dense"}:
        raise ContractError("density profiles must contain exactly low, medium, and dense")
    return rows


def owned_actor_cleanup_order(actors: Iterable[Mapping[str, Any]]) -> List[int]:
    """Return unique owned actor IDs in deterministic reverse spawn order."""
    normalized: List[Tuple[int, int]] = []
    seen: set[int] = set()
    for raw in actors:
        actor_id = raw.get("actor_id")
        spawn_index = raw.get("spawn_index")
        if not isinstance(actor_id, int) or actor_id <= 0 or not isinstance(spawn_index, int) or spawn_index < 0:
            raise ContractError("owned actors require positive actor_id and non-negative spawn_index")
        if actor_id in seen:
            raise ContractError(f"duplicate owned actor_id: {actor_id}")
        seen.add(actor_id)
        normalized.append((spawn_index, actor_id))
    normalized.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [actor_id for _, actor_id in normalized]


def _record_key(record: Mapping[str, Any], field: str) -> str:
    return _nonempty_string(record.get(field), f"record.{field}")


def _record_seed_bundle_id(record: Mapping[str, Any]) -> str:
    if isinstance(record.get("seed"), Mapping):
        return _nonempty_string(record["seed"].get("seed_bundle_id"), "record.seed.seed_bundle_id")
    return _record_key(record, "seed_bundle_id")


def split_episode_records(
    records: Sequence[Mapping[str, Any]], seed_partition: Mapping[str, Sequence[str]]
) -> Dict[str, Any]:
    """Assign complete episodes by frozen seed bundle and prove zero overlap."""
    if not records:
        raise ContractError("dataset record list is empty")
    unknown_splits = set(seed_partition) - set(SPLITS)
    if unknown_splits or set(seed_partition) != set(SPLITS):
        raise ContractError("seed partition must define train, validation, and final_test")
    bundle_to_split: Dict[str, str] = {}
    for split in SPLITS:
        for bundle in seed_partition[split]:
            bundle_id = _nonempty_string(bundle, f"seed_partition.{split}")
            if bundle_id in bundle_to_split:
                raise ContractError(f"seed bundle appears in multiple splits: {bundle_id}")
            bundle_to_split[bundle_id] = split

    episode_state: Dict[str, Dict[str, Any]] = {}
    split_frames: Dict[str, List[str]] = {split: [] for split in SPLITS}
    for record in records:
        episode_id = _record_key(record, "episode_id")
        frame_id = _record_key(record, "frame_id")
        seed_bundle_id = _record_seed_bundle_id(record)
        route_id = _record_key(record, "route_id")
        route_sha = _record_key(record, "route_sha256").lower()
        if not SHA256_RE.fullmatch(route_sha):
            raise ContractError(f"invalid route_sha256 in episode {episode_id}")
        if record.get("episode_complete") is not True and record.get("episode_status") != "COMPLETE":
            raise ContractError(f"incomplete episode cannot be split: {episode_id}")
        if seed_bundle_id not in bundle_to_split:
            raise ContractError(f"unassigned seed bundle: {seed_bundle_id}")
        split = bundle_to_split[seed_bundle_id]
        state = episode_state.setdefault(
            episode_id,
            {"seed_bundle_id": seed_bundle_id, "route_id": route_id, "route_sha256": route_sha, "split": split, "frame_ids": []},
        )
        invariant = (state["seed_bundle_id"], state["route_id"], state["route_sha256"], state["split"])
        observed = (seed_bundle_id, route_id, route_sha, split)
        if invariant != observed:
            raise ContractError(f"episode {episode_id} crosses seed, route, hash, or split boundaries")
        if frame_id in state["frame_ids"]:
            raise ContractError(f"duplicate frame_id {frame_id} in episode {episode_id}")
        state["frame_ids"].append(frame_id)
        split_frames[split].append(f"{episode_id}/{frame_id}")

    split_episodes: Dict[str, List[str]] = {
        split: sorted(ep for ep, state in episode_state.items() if state["split"] == split) for split in SPLITS
    }
    overlap: Dict[str, Dict[str, List[str]]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            label = f"{left}__{right}"
            left_eps, right_eps = set(split_episodes[left]), set(split_episodes[right])
            left_seeds = {episode_state[ep]["seed_bundle_id"] for ep in left_eps}
            right_seeds = {episode_state[ep]["seed_bundle_id"] for ep in right_eps}
            left_frames, right_frames = set(split_frames[left]), set(split_frames[right])
            overlap[label] = {
                "episode_ids": sorted(left_eps & right_eps),
                "seed_bundle_ids": sorted(left_seeds & right_seeds),
                "frame_keys": sorted(left_frames & right_frames),
            }
    leakage = any(values for pair in overlap.values() for values in pair.values())
    if leakage:
        raise ContractError("split leakage detected")
    return {
        "schema": "scenesense.episode_seed_split_manifest.v1",
        "strategy": "complete_episode_and_seed_bundle_holdout",
        "splits": {
            split: {
                "episode_ids": split_episodes[split],
                "seed_bundle_ids": sorted(seed_partition[split]),
                "frame_count": len(split_frames[split]),
                "frame_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(split_frames[split]))),
                "access_state": "LOCKED_UNTOUCHED" if split == "final_test" else "AVAILABLE",
            }
            for split in SPLITS
        },
        "overlap_checks": overlap,
        "leakage_detected": False,
        "final_test_evaluation_authorization_required": FINAL_TEST_MARKER,
    }


def split_route_region_records(
    records: Sequence[Mapping[str, Any]], region_partition: Mapping[str, Sequence[str]]
) -> Dict[str, Any]:
    """Alternative split: keep complete route regions together across splits."""
    if set(region_partition) != set(SPLITS):
        raise ContractError("region partition must define train, validation, and final_test")
    region_to_split: Dict[str, str] = {}
    for split in SPLITS:
        for region in region_partition[split]:
            region_id = _nonempty_string(region, f"region_partition.{split}")
            if region_id in region_to_split:
                raise ContractError(f"route region appears in multiple splits: {region_id}")
            region_to_split[region_id] = split
    groups: Dict[str, List[str]] = {split: [] for split in SPLITS}
    for record in records:
        region_id = _record_key(record, "route_region_id")
        if region_id not in region_to_split:
            raise ContractError(f"unassigned route region: {region_id}")
        groups[region_to_split[region_id]].append(f"{_record_key(record, 'episode_id')}/{_record_key(record, 'frame_id')}")
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if set(region_partition[left]) & set(region_partition[right]):
                raise ContractError("route-region leakage detected")
    return {
        "schema": "scenesense.route_region_split_manifest.v1",
        "strategy": "complete_route_region_holdout",
        "splits": {
            split: {
                "route_region_ids": sorted(region_partition[split]),
                "frame_count": len(groups[split]),
                "frame_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(groups[split]))),
                "access_state": "LOCKED_UNTOUCHED" if split == "final_test" else "AVAILABLE",
            }
            for split in SPLITS
        },
        "leakage_detected": False,
        "final_test_evaluation_authorization_required": FINAL_TEST_MARKER,
    }


def require_test_evaluation_authorization(
    marker_path: Path | str | None, *, dataset_manifest_sha256: str, pilot_manifest_sha256: str
) -> Mapping[str, Any]:
    if marker_path is None or not Path(marker_path).is_file():
        raise ContractError(f"final test evaluation is locked; missing {FINAL_TEST_MARKER}")
    marker = _load_document(Path(marker_path))
    _required(
        marker,
        ("authorization", "authorized_by", "authorized_at_utc", "dataset_manifest_sha256", "pilot_manifest_sha256"),
        "test authorization",
    )
    if marker["authorization"] != FINAL_TEST_MARKER:
        raise ContractError("invalid final-test authorization marker")
    if marker["dataset_manifest_sha256"] != dataset_manifest_sha256:
        raise ContractError("test authorization dataset hash mismatch")
    if marker["pilot_manifest_sha256"] != pilot_manifest_sha256:
        raise ContractError("test authorization pilot hash mismatch")
    return marker


def source_manifest(paths: Sequence[Path]) -> Dict[str, Any]:
    return {
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(paths, key=lambda item: str(item))
        ]
    }


def load_frame_records(path: Path | str) -> List[Mapping[str, Any]]:
    """Load either a JSON array or JSONL without altering any artifacts."""
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(f"frame-record input is missing: {source}") from exc
    try:
        if source.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid frame-record JSON: {exc}") from exc
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise ContractError("frame-record input must be a JSON array or JSONL stream of objects")
    return values


def emit_episode_seed_split_manifest(
    records_path: Path, collection_config_path: Path, output_directory: Path
) -> Tuple[Path, str]:
    records = load_frame_records(records_path)
    config = _load_document(collection_config_path)
    seed_bundles = _array(config.get("seed_bundles"), "collection.seed_bundles")
    partition: Dict[str, List[str]] = {split: [] for split in SPLITS}
    for index, raw in enumerate(seed_bundles):
        seed = _object(raw, f"seed_bundles[{index}]")
        _required(seed, ("seed_bundle_id", "reserved_split"), f"seed_bundles[{index}]")
        if seed["reserved_split"] not in SPLITS:
            raise ContractError(f"invalid reserved split: {seed['reserved_split']}")
        partition[str(seed["reserved_split"])].append(str(seed["seed_bundle_id"]))
    manifest = split_episode_records(records, partition)
    target = create_only_directory(output_directory)
    write_json_create_only(target / "SPLIT_MANIFEST.json", manifest)
    digest = sha256_file(target / "SPLIT_MANIFEST.json")
    write_create_only(target / "SPLIT_MANIFEST.sha256", (digest + "  SPLIT_MANIFEST.json\n").encode("ascii"))
    write_create_only(
        target / "REVIEW_REQUIRED",
        ("FINAL TEST REMAINS LOCKED_UNTOUCHED. " + FINAL_TEST_MARKER + " is required before evaluation.\n").encode("utf-8"),
    )
    return target, digest


def emit_readiness_bundle(
    output_root: Path,
    timestamp: str,
    collection_config_path: Path,
    pilot_config_path: Path,
    evaluation_config_path: Path,
    schema_paths: Sequence[Path],
    source_paths: Sequence[Path],
) -> Tuple[Path, str]:
    bundle = create_only_directory(output_root / f"{timestamp}_waiting")
    collection_config = _load_document(collection_config_path)
    matrix = build_dry_run_matrix(collection_config)
    preflight = {
        "schema": SCHEMA_VERSION,
        "terminal": TERMINAL_WAITING,
        "carla_imported": False,
        "carla_start_attempted": False,
        "collection_started": False,
        "training_started": False,
        "validation_evaluation_started": False,
        "final_test_evaluation_started": False,
        "reason": "Qualified canonical Town10HD Route B file and external SHA-256 have not been supplied.",
        "required_before_collection_preflight_can_pass": [
            "canonical_route_b.json|yaml",
            "external lowercase SHA-256 digest of the exact route file bytes",
            "route qualification evidence fields embedded in the route contract",
        ],
    }
    plan = {
        "schema": SCHEMA_VERSION,
        "objective": "Route-independent full-map perception collection scaffolding and bounded AE64-first retraining pilot",
        "terminal": TERMINAL_WAITING,
        "authorized_now": ["offline contract validation", "dry-run matrix generation", "leakage-safe splitting", "unit tests"],
        "prohibited_now": [
            "CARLA startup or collection",
            "model training",
            "untouched final-test evaluation",
            "production runtime, registry, launcher, controller, or map-server edits",
            "checkpoint overwrite",
        ],
        "gates": [
            "qualified canonical Route B and matching SHA-256",
            "create-only collection output",
            "complete-episode/seed or complete-route-region splits",
            "AE64 validation feasibility and selection evidence",
            FINAL_TEST_MARKER,
        ],
    }
    route_requirements = {
        "required_files_from_main_machine": [
            {
                "file": "canonical_route_b.json (preferred), canonical_route_b.yaml, or canonical_route_b.yml",
                "content": "A contract conforming to schemas/canonical_route_v1.schema.json and the semantic validator",
            },
            {
                "file": "canonical_route_b.sha256 or an equivalent separately transmitted digest",
                "content": "Exactly one lowercase SHA-256 digest for the exact route-file bytes",
            },
        ],
        "required_route_fields": [
            "schema=scenesense.canonical_route.v1",
            "route_id identifying Route B",
            "version (semantic version)",
            "map.town=Town10HD and map.map_identifier",
            "coordinate_frame name/units/handedness/x-y-z axes",
            "route_direction.direction and waypoint_order=travel_order",
            "ordered waypoints with sequence_index/x/y/z/route_segment_id",
            "contiguous segments with IDs and waypoint ranges",
            "ego_spawn_transform location/rotation/start waypoint/tolerance",
            "loop_closure mode/close flag/position tolerance/heading tolerance",
            "route_length declared value/calculation/tolerance",
            "expected_duration minimum/nominal/maximum seconds",
            "qualification status/by/time/bundle ID/qualification manifest SHA-256",
        ],
        "coordinates": "No coordinates are included in this scaffold; they must come from the qualified main-machine artifact.",
    }
    report = {
        "terminal": TERMINAL_WAITING,
        "readiness": "NOT_READY_FOR_COLLECTION",
        "existing_decision": "RETRAINING_PILOT_JUSTIFIED",
        "scope": "Bounded AE64-first pilot only; no family-wide retraining and no deployment approval.",
        "dry_run_episode_count": len(matrix),
        "dry_run_dimensions": ["density_profile", "seed_bundle"],
        "excluded_matrix_dimensions": ["UE action", "network profile", "model family", "quantization", "ROI"],
        "final_test_state": "LOCKED_UNTOUCHED",
    }
    seed_partition = {split: [] for split in SPLITS}
    for seed in collection_config["seed_bundles"]:
        seed_partition[seed["reserved_split"]].append(seed["seed_bundle_id"])
    split_policy = {
        "schema": "scenesense.split_policy_freeze.v1",
        "default_strategy": "complete_episode_and_seed_bundle_holdout",
        "alternative_strategy": "complete_route_region_holdout",
        "random_individual_frame_split_allowed": False,
        "seed_partition": seed_partition,
        "overlap_axes_required": ["episode_id", "seed_bundle_id", "frame_key"],
        "current_overlap_check_status": "NOT_RUN_NO_COLLECTED_DATASET",
        "final_test_access_state": "LOCKED_UNTOUCHED",
        "final_test_required_marker": FINAL_TEST_MARKER,
    }
    artifacts: Dict[str, bytes] = {
        "PLAN.json": pretty_json_bytes(plan),
        "PREFLIGHT.json": pretty_json_bytes(preflight),
        "DRY_RUN_MATRIX.json": pretty_json_bytes({"schema": SCHEMA_VERSION, "rows": matrix}),
        "ROUTE_B_REQUIREMENTS.json": pretty_json_bytes(route_requirements),
        "REPORT.json": pretty_json_bytes(report),
        "DATASET_SPLIT_POLICY.json": pretty_json_bytes(split_policy),
        "SOURCE_PROVENANCE.json": pretty_json_bytes(source_manifest(list(source_paths) + [collection_config_path, pilot_config_path, evaluation_config_path] + list(schema_paths))),
        "REVIEW_REQUIRED": (TERMINAL_WAITING + "\nNo collection, training, final-test evaluation, promotion, or deployment is authorized.\n").encode("utf-8"),
    }
    for name, payload in artifacts.items():
        write_create_only(bundle / name, payload)
    manifest = {
        "schema": "scenesense.immutable_readiness_manifest.v1",
        "created_at_utc": utc_now(),
        "terminal": TERMINAL_WAITING,
        "bundle_name": bundle.name,
        "artifacts": [
            {"path": name, "sha256": sha256_file(bundle / name), "size_bytes": (bundle / name).stat().st_size}
            for name in sorted(artifacts)
        ],
    }
    write_json_create_only(bundle / "MANIFEST.json", manifest)
    manifest_sha = sha256_file(bundle / "MANIFEST.json")
    write_create_only(bundle / "MANIFEST.sha256", (manifest_sha + "  MANIFEST.json\n").encode("ascii"))
    os.chmod(bundle, 0o555)
    return bundle, manifest_sha


def _project_paths() -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    workspace = here.parents[1]
    schemas = here / "schemas"
    return {
        "collection": here / "configs" / "perception_full_route_collection_v1.json",
        "pilot": here / "configs" / "perception_ae64_pilot_v1.json",
        "evaluation": here / "configs" / "perception_full_route_evaluation_v1.json",
        "schemas": [
            schemas / "canonical_route_v1.schema.json",
            schemas / "perception_frame_v1.schema.json",
            schemas / "perception_episode_v1.schema.json",
            schemas / "perception_split_input_v1.schema.json",
            schemas / "test_evaluation_authorization_v1.schema.json",
        ],
        "sources": [
            here / "perception_full_route_pilot_v1.py",
            workspace / "abiodun" / "carla_collect_moving_ego_fusion_training_data.py",
            workspace / "abiodun" / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion" / "object_targets.py",
            workspace / "abiodun" / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion" / "train_fusion.py",
            workspace / "abiodun" / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion" / "evaluate_fusion.py",
            workspace / "abiodun" / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion" / "split_runtime.py",
            workspace / "abiodun" / "experiments" / "ae_integrated_20260710" / "ae64" / "checkpoints" / "ae64_integrated" / "best.pt",
            here / "experiments" / "conservative_decoder_validation_v1" / "20260820_233246_EDT" / "PREINFERENCE_FREEZE.json",
            here / "experiments" / "conservative_decoder_validation_v1" / "20260820_233246_EDT" / "RETRAINING_PILOT_PROPOSAL.md",
        ],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("route-preflight", help="hash and validate a canonical Route B artifact")
    preflight.add_argument("--route-file")
    preflight.add_argument("--route-sha256")
    matrix = subparsers.add_parser("dry-run-matrix", help="create an offline density-by-seed collection matrix")
    matrix.add_argument("--output", required=True, type=Path)
    split = subparsers.add_parser("split-manifest", help="create a complete-episode/seed split manifest")
    split.add_argument("--records", required=True, type=Path, help="JSON array or JSONL frame records")
    split.add_argument("--output-directory", required=True, type=Path)
    split.add_argument(
        "--collection-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "perception_full_route_collection_v1.json",
    )
    readiness = subparsers.add_parser("readiness", help="emit an immutable waiting readiness bundle")
    readiness.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments" / "perception_full_route_ae64_scaffold_v1",
    )
    readiness.add_argument("--timestamp", required=True, help="Caller-frozen UTC/local timestamp token")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "route-preflight":
        try:
            _, summary = verify_canonical_route(args.route_file, args.route_sha256)
        except ContractError as exc:
            print(json.dumps({"terminal": TERMINAL_WAITING, "error": str(exc)}, sort_keys=True))
            return 3
        print(json.dumps(summary, sort_keys=True))
        return 0
    paths = _project_paths()
    if args.command == "dry-run-matrix":
        try:
            rows = build_dry_run_matrix(_load_document(paths["collection"]))
            write_json_create_only(args.output, {"schema": SCHEMA_VERSION, "rows": rows, "terminal": TERMINAL_WAITING})
        except ContractError as exc:
            print(json.dumps({"terminal": "SCAFFOLD_ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return 2
        print(json.dumps({"terminal": TERMINAL_WAITING, "matrix": str(args.output.resolve()), "row_count": len(rows)}, sort_keys=True))
        return 0
    if args.command == "split-manifest":
        try:
            output, digest = emit_episode_seed_split_manifest(args.records, args.collection_config, args.output_directory)
        except ContractError as exc:
            print(json.dumps({"terminal": "SCAFFOLD_ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return 2
        print(json.dumps({"terminal": TERMINAL_WAITING, "split_manifest_directory": str(output.resolve()), "manifest_sha256": digest}, sort_keys=True))
        return 0
    try:
        bundle, manifest_sha = emit_readiness_bundle(
            args.output_root,
            args.timestamp,
            paths["collection"],
            paths["pilot"],
            paths["evaluation"],
            paths["schemas"],
            paths["sources"],
        )
    except ContractError as exc:
        print(json.dumps({"terminal": "SCAFFOLD_ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"terminal": TERMINAL_WAITING, "bundle": str(bundle.resolve()), "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

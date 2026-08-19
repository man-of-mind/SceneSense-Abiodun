"""Create-only static environment truth for future Phase-2 captures.

This module snapshots CARLA map environment objects only.  Dynamic actors stay
in the existing per-frame actor-origin truth stream.  The implementation is
duck-typed so its contract can be unit-tested without importing or launching
CARLA.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence


STATIC_TRUTH_SCHEMA = "scenesense.phase2_static_environment_truth.v1"
STATIC_MANIFEST_SCHEMA = "scenesense.phase2_static_environment_artifacts.v1"
OBJECTS_CSV_NAME = "static_environment_objects.csv"
SNAPSHOT_JSON_NAME = "static_environment_snapshot.json"
MANIFEST_JSON_NAME = "artifact_manifest.json"
MAP_HASH_BASIS = "sha256_utf8_carla_map_to_opendrive"
BBOX_COORDINATE_FRAME = "carla_world_as_returned"
EXPECTED_FILES = frozenset(
    {OBJECTS_CSV_NAME, SNAPSHOT_JSON_NAME, MANIFEST_JSON_NAME}
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

OBJECT_FIELDS = (
    "schema",
    "environment_object_id",
    "carla_environment_object_id",
    "object_name",
    "semantic_class",
    "enabled",
    "transform_x_m",
    "transform_y_m",
    "transform_z_m",
    "transform_roll_deg",
    "transform_pitch_deg",
    "transform_yaw_deg",
    "bbox_center_x_m",
    "bbox_center_y_m",
    "bbox_center_z_m",
    "bbox_extent_x_m",
    "bbox_extent_y_m",
    "bbox_extent_z_m",
    "bbox_rotation_roll_deg",
    "bbox_rotation_pitch_deg",
    "bbox_rotation_yaw_deg",
    "bbox_coordinate_frame",
    "map_name",
    "map_sha256",
    "capture_clock_id",
    "capture_frame_id",
    "capture_timestamp_s",
)

FLOAT_FIELDS = (
    "transform_x_m",
    "transform_y_m",
    "transform_z_m",
    "transform_roll_deg",
    "transform_pitch_deg",
    "transform_yaw_deg",
    "bbox_center_x_m",
    "bbox_center_y_m",
    "bbox_center_z_m",
    "bbox_extent_x_m",
    "bbox_extent_y_m",
    "bbox_extent_z_m",
    "bbox_rotation_roll_deg",
    "bbox_rotation_pitch_deg",
    "bbox_rotation_yaw_deg",
    "capture_timestamp_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json_x(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _label_name(value: object) -> str:
    explicit = getattr(value, "name", None)
    text = str(explicit if explicit is not None else value).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if not text:
        raise ValueError("semantic label must be non-empty")
    return text


def _native_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("environment-object native ID must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("environment-object native ID must be an integer") from exc
    if result < 0:
        raise ValueError("environment-object native ID must be non-negative")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(result)
    if not math.isfinite(numeric) or numeric != float(result):
        raise ValueError("environment-object native ID must be an exact integer")
    return result


def stable_environment_object_id_v1(map_sha256: str, native_id: int) -> str:
    """Return a deterministic map-scoped ID without using pose or object name."""

    digest_text = str(map_sha256).lower()
    if SHA256_PATTERN.fullmatch(digest_text) is None:
        raise ValueError("map_sha256 must be a lowercase SHA-256 digest")
    native = _native_id(native_id)
    identity = f"{digest_text}\x00{native}".encode("utf-8")
    return f"envobj:v1:{hashlib.sha256(identity).hexdigest()}"


def _transform_values(transform: object, prefix: str) -> dict[str, float]:
    try:
        location = transform.location
        rotation = transform.rotation
    except AttributeError as exc:
        raise ValueError(f"{prefix} lacks CARLA transform fields") from exc
    return {
        f"{prefix}_x_m": _finite(location.x, f"{prefix}.location.x"),
        f"{prefix}_y_m": _finite(location.y, f"{prefix}.location.y"),
        f"{prefix}_z_m": _finite(location.z, f"{prefix}.location.z"),
        f"{prefix}_roll_deg": _finite(rotation.roll, f"{prefix}.rotation.roll"),
        f"{prefix}_pitch_deg": _finite(
            rotation.pitch, f"{prefix}.rotation.pitch"
        ),
        f"{prefix}_yaw_deg": _finite(rotation.yaw, f"{prefix}.rotation.yaw"),
    }


def _bounding_box_values(bounding_box: object) -> dict[str, float]:
    try:
        center = bounding_box.location
        extent = bounding_box.extent
        rotation = bounding_box.rotation
    except AttributeError as exc:
        raise ValueError("environment-object bounding box is incomplete") from exc
    values = {
        "bbox_center_x_m": _finite(center.x, "bounding_box.location.x"),
        "bbox_center_y_m": _finite(center.y, "bounding_box.location.y"),
        "bbox_center_z_m": _finite(center.z, "bounding_box.location.z"),
        "bbox_extent_x_m": _finite(extent.x, "bounding_box.extent.x"),
        "bbox_extent_y_m": _finite(extent.y, "bounding_box.extent.y"),
        "bbox_extent_z_m": _finite(extent.z, "bounding_box.extent.z"),
        "bbox_rotation_roll_deg": _finite(
            rotation.roll, "bounding_box.rotation.roll"
        ),
        "bbox_rotation_pitch_deg": _finite(
            rotation.pitch, "bounding_box.rotation.pitch"
        ),
        "bbox_rotation_yaw_deg": _finite(
            rotation.yaw, "bounding_box.rotation.yaw"
        ),
    }
    if any(values[field] <= 0.0 for field in (
        "bbox_extent_x_m",
        "bbox_extent_y_m",
        "bbox_extent_z_m",
    )):
        raise ValueError("oriented bounding-box extents must all be positive")
    return values


def _enabled_registry(values: Mapping[object, object]) -> dict[int, bool]:
    if not isinstance(values, Mapping):
        raise ValueError("enabled_state_by_id must be a mapping")
    normalized: dict[int, bool] = {}
    for key, value in values.items():
        native = _native_id(key)
        if native in normalized:
            raise ValueError(f"duplicate enabled-state ID: {native}")
        if type(value) is not bool:
            raise ValueError("every enabled state must be a boolean")
        normalized[native] = value
    return normalized


def _map_contract(world: object) -> tuple[object, str, str, int]:
    try:
        carla_map = world.get_map()
        map_name = str(carla_map.name).strip()
        opendrive = carla_map.to_opendrive()
    except AttributeError as exc:
        raise ValueError("world does not expose the CARLA map contract") from exc
    if not map_name:
        raise ValueError("CARLA map name must be non-empty")
    if not isinstance(opendrive, str) or not opendrive:
        raise ValueError("CARLA map OpenDRIVE text must be non-empty")
    encoded = opendrive.encode("utf-8")
    return carla_map, map_name, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _capture_clock(world: object) -> tuple[int, float]:
    try:
        snapshot = world.get_snapshot()
        frame = _native_id(snapshot.frame)
        timestamp = _finite(
            snapshot.timestamp.elapsed_seconds,
            "world_snapshot.timestamp.elapsed_seconds",
        )
    except AttributeError as exc:
        raise ValueError("world does not expose a capture snapshot clock") from exc
    if timestamp < 0.0:
        raise ValueError("capture timestamp must be non-negative")
    return frame, timestamp


def _object_row(
    environment_object: object,
    *,
    map_name: str,
    map_sha256: str,
    capture_frame_id: int,
    capture_timestamp_s: float,
    enabled: bool,
) -> dict[str, object]:
    try:
        native_id = _native_id(environment_object.id)
        semantic_class = _label_name(environment_object.type)
        object_name = str(environment_object.name)
        transform = environment_object.transform
        bounding_box = environment_object.bounding_box
    except AttributeError as exc:
        raise ValueError("environment object lacks required CARLA fields") from exc
    return {
        "schema": STATIC_TRUTH_SCHEMA,
        "environment_object_id": stable_environment_object_id_v1(
            map_sha256, native_id
        ),
        "carla_environment_object_id": native_id,
        "object_name": object_name,
        "semantic_class": semantic_class,
        "enabled": bool(enabled),
        **_transform_values(transform, "transform"),
        **_bounding_box_values(bounding_box),
        "bbox_coordinate_frame": BBOX_COORDINATE_FRAME,
        "map_name": map_name,
        "map_sha256": map_sha256,
        "capture_clock_id": "carla_sim_clock",
        "capture_frame_id": capture_frame_id,
        "capture_timestamp_s": capture_timestamp_s,
    }


def _serialized_row(row: Mapping[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for field in OBJECT_FIELDS:
        value = row[field]
        if field in FLOAT_FIELDS:
            serialized[field] = _format_float(float(value))
        elif field == "enabled":
            serialized[field] = "true" if bool(value) else "false"
        else:
            serialized[field] = value
    return serialized


def capture_static_environment_truth_v1(
    world: object,
    output_dir: Path,
    *,
    semantic_labels: Sequence[object],
    required_semantic_classes: Sequence[object],
    enabled_state_by_id: Mapping[object, object],
    selection_contract: str,
) -> dict:
    """Snapshot selected static map objects and seal a create-only artifact."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    selection = str(selection_contract).strip()
    if not selection:
        raise ValueError("selection_contract must be non-empty")
    labels = list(semantic_labels)
    if not labels:
        raise ValueError("semantic_labels must not be empty")
    label_names = [_label_name(label) for label in labels]
    if len(label_names) != len(set(label_names)):
        raise ValueError("semantic_labels contain duplicates")
    required = [_label_name(value) for value in required_semantic_classes]
    if not required or len(required) != len(set(required)):
        raise ValueError("required_semantic_classes must be unique and non-empty")
    enabled_registry = _enabled_registry(enabled_state_by_id)
    _, map_name, map_sha256, opendrive_bytes = _map_contract(world)
    capture_frame_id, capture_timestamp_s = _capture_clock(world)

    rows: list[dict[str, object]] = []
    seen_native_ids: set[int] = set()
    for label in labels:
        try:
            objects = world.get_environment_objects(label)
        except AttributeError as exc:
            raise ValueError(
                "world does not expose get_environment_objects"
            ) from exc
        if objects is None:
            raise ValueError("get_environment_objects returned None")
        for environment_object in objects:
            try:
                native_id = _native_id(environment_object.id)
            except AttributeError as exc:
                raise ValueError("environment object lacks native ID") from exc
            if native_id in seen_native_ids:
                raise ValueError(
                    f"duplicate environment-object native ID across queries: {native_id}"
                )
            seen_native_ids.add(native_id)
            if native_id not in enabled_registry:
                raise ValueError(
                    f"enabled state is missing for environment object {native_id}"
                )
            rows.append(
                _object_row(
                    environment_object,
                    map_name=map_name,
                    map_sha256=map_sha256,
                    capture_frame_id=capture_frame_id,
                    capture_timestamp_s=capture_timestamp_s,
                    enabled=enabled_registry[native_id],
                )
            )
    if not rows:
        raise ValueError("static environment snapshot must contain at least one object")
    rows.sort(key=lambda row: int(row["carla_environment_object_id"]))
    class_counts = Counter(str(row["semantic_class"]) for row in rows)
    missing_classes = sorted(set(required) - set(class_counts))
    if missing_classes:
        raise ValueError(
            f"required semantic classes are absent: {missing_classes}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    csv_path = output / OBJECTS_CSV_NAME
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OBJECT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_serialized_row(row))

    snapshot_path = output / SNAPSHOT_JSON_NAME
    _write_json_x(
        snapshot_path,
        {
            "schema": STATIC_TRUTH_SCHEMA,
            "status": "complete",
            "catalog_scope": "static_carla_environment_objects_only",
            "source_api": "world.get_environment_objects",
            "selection_contract": selection,
            "requested_semantic_labels": label_names,
            "required_semantic_classes": required,
            "map": {
                "name": map_name,
                "sha256": map_sha256,
                "hash_basis": MAP_HASH_BASIS,
                "opendrive_utf8_bytes": opendrive_bytes,
            },
            "capture": {
                "clock_id": "carla_sim_clock",
                "frame_id": capture_frame_id,
                "timestamp_s": capture_timestamp_s,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            },
            "enabled_state_source": "capture_owner_explicit_registry",
            "dynamic_actor_truth": {
                "included": False,
                "contract": "separate_per_frame_actor_origin_stream",
            },
            "object_count": len(rows),
            "semantic_class_counts": dict(sorted(class_counts.items())),
            "objects_csv": {
                "path": OBJECTS_CSV_NAME,
                "bytes": csv_path.stat().st_size,
                "sha256": _sha256(csv_path),
                "columns": list(OBJECT_FIELDS),
            },
        },
    )

    manifest_path = output / MANIFEST_JSON_NAME
    manifest_files = []
    for path in (csv_path, snapshot_path):
        manifest_files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json_x(
        manifest_path,
        {
            "schema": STATIC_MANIFEST_SCHEMA,
            "files": manifest_files,
        },
    )
    return verify_static_environment_truth_v1(output)


def _verify_manifest(output: Path) -> dict:
    entries = list(output.iterdir())
    actual_names = {path.name for path in entries}
    if (
        actual_names != EXPECTED_FILES
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError(
            "static truth directory has missing or unexpected files: "
            f"expected={sorted(EXPECTED_FILES)} actual={sorted(actual_names)}"
        )
    manifest = _json(output / MANIFEST_JSON_NAME)
    if manifest.get("schema") != STATIC_MANIFEST_SCHEMA:
        raise ValueError("static truth artifact manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("static truth artifact manifest must list exactly two files")
    records: dict[str, dict] = {}
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("artifact manifest file record is invalid")
        path_name = str(record.get("path", ""))
        if path_name not in {OBJECTS_CSV_NAME, SNAPSHOT_JSON_NAME}:
            raise ValueError("artifact manifest contains an unexpected path")
        if path_name in records:
            raise ValueError("artifact manifest contains a duplicate path")
        path = output / path_name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or _sha256(path) != str(record.get("sha256", ""))
        ):
            raise ValueError(f"static truth artifact integrity mismatch: {path_name}")
        records[path_name] = record
    if set(records) != {OBJECTS_CSV_NAME, SNAPSHOT_JSON_NAME}:
        raise ValueError("artifact manifest is incomplete")
    return manifest


def verify_static_environment_truth_v1(
    output_dir: Path,
    *,
    expected_map_name: Optional[str] = None,
    expected_map_sha256: Optional[str] = None,
    expected_capture_frame_id: Optional[int] = None,
    expected_selection_contract: Optional[str] = None,
    expected_required_semantic_classes: Optional[Sequence[object]] = None,
) -> dict:
    """Fail closed on any incomplete, inconsistent, or modified snapshot."""

    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(output)
    _verify_manifest(output)
    snapshot = _json(output / SNAPSHOT_JSON_NAME)
    if snapshot.get("schema") != STATIC_TRUTH_SCHEMA:
        raise ValueError("static environment snapshot schema mismatch")
    if snapshot.get("status") != "complete":
        raise ValueError("static environment snapshot is incomplete")
    if snapshot.get("catalog_scope") != "static_carla_environment_objects_only":
        raise ValueError("static environment snapshot scope mismatch")
    if snapshot.get("source_api") != "world.get_environment_objects":
        raise ValueError("static environment source API mismatch")
    dynamic = snapshot.get("dynamic_actor_truth")
    if not isinstance(dynamic, dict) or dynamic != {
        "included": False,
        "contract": "separate_per_frame_actor_origin_stream",
    }:
        raise ValueError("dynamic actor truth must remain explicitly separate")

    map_record = snapshot.get("map")
    capture = snapshot.get("capture")
    objects_csv = snapshot.get("objects_csv")
    if not all(isinstance(value, dict) for value in (map_record, capture, objects_csv)):
        raise ValueError("snapshot map, capture, and CSV records are required")
    map_name = str(map_record.get("name", ""))
    map_sha256 = str(map_record.get("sha256", ""))
    if not map_name or SHA256_PATTERN.fullmatch(map_sha256) is None:
        raise ValueError("snapshot map identity is invalid")
    if map_record.get("hash_basis") != MAP_HASH_BASIS:
        raise ValueError("snapshot map hash basis mismatch")
    if int(map_record.get("opendrive_utf8_bytes", 0)) <= 0:
        raise ValueError("snapshot map OpenDRIVE byte count is invalid")
    if expected_map_name is not None and map_name != str(expected_map_name):
        raise ValueError("snapshot map name differs from expected map name")
    if (
        expected_map_sha256 is not None
        and map_sha256 != str(expected_map_sha256)
    ):
        raise ValueError("snapshot map hash differs from expected map hash")

    capture_frame = _native_id(capture.get("frame_id"))
    capture_timestamp = _finite(capture.get("timestamp_s"), "capture.timestamp_s")
    if capture.get("clock_id") != "carla_sim_clock" or capture_timestamp < 0.0:
        raise ValueError("snapshot capture clock is invalid")
    if (
        expected_capture_frame_id is not None
        and capture_frame != int(expected_capture_frame_id)
    ):
        raise ValueError("snapshot capture frame differs from expected frame")
    if not str(capture.get("written_utc", "")):
        raise ValueError("snapshot wall-clock write timestamp is missing")

    csv_path = output / OBJECTS_CSV_NAME
    if (
        objects_csv.get("path") != OBJECTS_CSV_NAME
        or int(objects_csv.get("bytes", -1)) != csv_path.stat().st_size
        or str(objects_csv.get("sha256", "")) != _sha256(csv_path)
        or objects_csv.get("columns") != list(OBJECT_FIELDS)
    ):
        raise ValueError("snapshot CSV integrity metadata mismatch")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != OBJECT_FIELDS:
            raise ValueError("static environment CSV columns differ from schema")
        rows = list(reader)
    expected_count = int(snapshot.get("object_count", -1))
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError("static environment CSV object count mismatch")

    native_ids: list[int] = []
    stable_ids: set[str] = set()
    class_counts: Counter[str] = Counter()
    for row in rows:
        if row["schema"] != STATIC_TRUTH_SCHEMA:
            raise ValueError("static environment CSV row schema mismatch")
        native_id = _native_id(row["carla_environment_object_id"])
        native_ids.append(native_id)
        stable_id = row["environment_object_id"]
        if stable_id != stable_environment_object_id_v1(map_sha256, native_id):
            raise ValueError("stable environment-object ID mismatch")
        if stable_id in stable_ids:
            raise ValueError("duplicate stable environment-object ID")
        stable_ids.add(stable_id)
        semantic_class = row["semantic_class"].strip()
        if not semantic_class:
            raise ValueError("static environment semantic class is empty")
        class_counts[semantic_class] += 1
        if row["enabled"] not in {"true", "false"}:
            raise ValueError("static environment enabled state is invalid")
        if row["bbox_coordinate_frame"] != BBOX_COORDINATE_FRAME:
            raise ValueError("static environment bbox coordinate frame mismatch")
        if row["map_name"] != map_name or row["map_sha256"] != map_sha256:
            raise ValueError("static environment row map identity mismatch")
        if row["capture_clock_id"] != "carla_sim_clock":
            raise ValueError("static environment row clock mismatch")
        if _native_id(row["capture_frame_id"]) != capture_frame:
            raise ValueError("static environment row capture frame mismatch")
        for field in FLOAT_FIELDS:
            value = _finite(row[field], field)
            if field.startswith("bbox_extent_") and value <= 0.0:
                raise ValueError("oriented bounding-box extents must all be positive")
        if not math.isclose(
            float(row["capture_timestamp_s"]),
            capture_timestamp,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("static environment row capture timestamp mismatch")
    if native_ids != sorted(native_ids) or len(native_ids) != len(set(native_ids)):
        raise ValueError("static environment native IDs must be unique and sorted")
    if dict(sorted(class_counts.items())) != snapshot.get("semantic_class_counts"):
        raise ValueError("static environment semantic-class counts mismatch")
    required = snapshot.get("required_semantic_classes")
    requested = snapshot.get("requested_semantic_labels")
    if (
        not isinstance(required, list)
        or not required
        or len(required) != len(set(required))
        or not set(required).issubset(class_counts)
    ):
        raise ValueError("required static semantic-class contract is not satisfied")
    if (
        not isinstance(requested, list)
        or not requested
        or len(requested) != len(set(requested))
    ):
        raise ValueError("requested static semantic-label contract is invalid")
    if snapshot.get("enabled_state_source") != "capture_owner_explicit_registry":
        raise ValueError("enabled-state provenance is invalid")
    selection_contract = str(snapshot.get("selection_contract", "")).strip()
    if not selection_contract:
        raise ValueError("static environment selection contract is missing")
    if (
        expected_selection_contract is not None
        and selection_contract != str(expected_selection_contract).strip()
    ):
        raise ValueError("static environment selection contract differs from expected")
    if expected_required_semantic_classes is not None:
        expected_required = {
            _label_name(value) for value in expected_required_semantic_classes
        }
        if expected_required != set(required):
            raise ValueError(
                "required static semantic classes differ from expected contract"
            )

    return {
        "schema": STATIC_TRUTH_SCHEMA,
        "verdict": "PASS",
        "output_dir": str(output.resolve()),
        "object_count": len(rows),
        "map_name": map_name,
        "map_sha256": map_sha256,
        "capture_frame_id": capture_frame,
        "capture_timestamp_s": capture_timestamp,
        "semantic_class_counts": dict(sorted(class_counts.items())),
        "dynamic_actor_truth_separate": True,
    }

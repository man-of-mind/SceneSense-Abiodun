"""Pure data contracts for timestamped object updates from multiple UEs.

This module intentionally performs no association or fusion. Its job is to
make source identity, capture time, coordinate values, and provenance explicit
before observations from different vehicles can meet in one spatial map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SPLITFUSION_EDGE_RESULT_SCHEMA = "splitfusion_edge_result.v2"
SPLITFUSION_OBJECT_UPDATE_SCHEMA = "splitfusion_object_map_update.v1"
LEGACY_SPATIAL_PACKET_SCHEMA = "fusion_object_spatial_map.v1"


class MultiUEContractError(ValueError):
    """Raised when an update cannot safely enter the cooperative map."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiUEContractError(message)


def _nonempty(value: object, label: str) -> str:
    result = str(value or "").strip()
    _require(bool(result), f"{label} must not be empty")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MultiUEContractError(f"{label} must be an integer") from exc
    _require(result >= minimum, f"{label} must be >= {minimum}")
    return result


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MultiUEContractError(f"{label} must be finite") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise MultiUEContractError("source payload is not canonical JSON") from exc


@dataclass(frozen=True)
class ObjectObservation:
    """One immutable world-frame object observation from one source frame."""

    observation_id: str
    class_name: str
    score: float
    world_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_deg: float
    raw_record_json: str
    raw_record_sha256: str


@dataclass(frozen=True)
class ObservationBatch:
    """One captured frame of compact object observations from one UE."""

    ue_id: str
    stream_id: str
    session_id: str
    clock_domain: str
    frame_id: int
    capture_timestamp_ns: int
    received_clock_domain: str
    received_timestamp_ns: int
    action_id: int | None
    source_schema: str
    source_payload_sha256: str
    objects: tuple[ObjectObservation, ...]

    @property
    def source_key(self) -> tuple[str, str, str]:
        return self.ue_id, self.session_id, self.stream_id

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (*self.source_key, self.frame_id)


def _normalise_record(record: Mapping[str, Any], index: int) -> ObjectObservation:
    location = record.get("location")
    if not isinstance(location, Mapping):
        location = {}
    dimensions = record.get("dimensions")
    if not isinstance(dimensions, Mapping):
        dimensions = {}

    world_xyz = (
        _finite(record.get("world_x", location.get("x")), "world_x"),
        _finite(record.get("world_y", location.get("y")), "world_y"),
        _finite(record.get("world_z", location.get("z", 0.0)), "world_z"),
    )
    size_lwh = (
        _finite(record.get("size_x", dimensions.get("length", 0.05)), "size_x"),
        _finite(record.get("size_y", dimensions.get("width", 0.05)), "size_y"),
        _finite(record.get("size_z", dimensions.get("height", 0.05)), "size_z"),
    )
    _require(all(value > 0.0 for value in size_lwh), "object dimensions must be positive")

    if "yaw_deg" in record:
        yaw_deg = _finite(record["yaw_deg"], "yaw_deg")
    elif "yaw_sin" in record and "yaw_cos" in record:
        yaw_deg = math.degrees(
            math.atan2(
                _finite(record["yaw_sin"], "yaw_sin"),
                _finite(record["yaw_cos"], "yaw_cos"),
            )
        )
    else:
        yaw_deg = 0.0

    score = _finite(record.get("score"), "score")
    _require(0.0 <= score <= 1.0, "score must lie in [0, 1]")
    class_name = _nonempty(record.get("class_name", record.get("type")), "class_name")
    observation_id = _nonempty(
        record.get("candidate_identity", record.get("id", f"observation_{index}")),
        "observation_id",
    )
    raw_record_json = _canonical_json(dict(record))
    return ObjectObservation(
        observation_id=observation_id,
        class_name=class_name,
        score=score,
        world_xyz=world_xyz,
        size_lwh=size_lwh,
        yaw_deg=yaw_deg,
        raw_record_json=raw_record_json,
        raw_record_sha256=hashlib.sha256(raw_record_json.encode("utf-8")).hexdigest(),
    )


def _batch(
    *,
    ue_id: object,
    stream_id: object,
    session_id: object,
    clock_domain: object,
    frame_id: object,
    capture_timestamp_ns: object,
    received_clock_domain: object,
    received_timestamp_ns: object,
    action_id: object | None,
    source_schema: str,
    source_payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> ObservationBatch:
    capture_ns = _integer(capture_timestamp_ns, "capture_timestamp_ns")
    received_ns = _integer(received_timestamp_ns, "received_timestamp_ns")
    capture_domain = _nonempty(clock_domain, "clock_domain")
    receipt_domain = _nonempty(received_clock_domain, "received_clock_domain")
    if receipt_domain == capture_domain:
        _require(received_ns >= capture_ns, "received timestamp precedes capture timestamp")
    payload_json = _canonical_json(dict(source_payload))
    objects = tuple(_normalise_record(record, index) for index, record in enumerate(records))
    return ObservationBatch(
        ue_id=_nonempty(ue_id, "ue_id"),
        stream_id=_nonempty(stream_id, "stream_id"),
        session_id=_nonempty(session_id, "session_id"),
        clock_domain=capture_domain,
        frame_id=_integer(frame_id, "frame_id"),
        capture_timestamp_ns=capture_ns,
        received_clock_domain=receipt_domain,
        received_timestamp_ns=received_ns,
        action_id=None if action_id is None else _integer(action_id, "action_id"),
        source_schema=source_schema,
        source_payload_sha256=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        objects=objects,
    )


def from_splitfusion_edge_result(
    payload: Mapping[str, Any],
    *,
    ue_id: str,
    session_id: str,
    received_timestamp_ns: int,
    clock_domain: str = "unix_wall_ns",
) -> ObservationBatch:
    """Validate the compact Phase-15 edge result as one UE observation batch."""

    _require(
        payload.get("schema") == SPLITFUSION_EDGE_RESULT_SCHEMA,
        "unexpected SplitFusion edge-result schema",
    )
    update = payload.get("object_map_update")
    _require(isinstance(update, Mapping), "edge result lacks object_map_update")
    _require(
        update.get("schema") == SPLITFUSION_OBJECT_UPDATE_SCHEMA,
        "unexpected SplitFusion object-update schema",
    )
    stream_id = _nonempty(payload.get("stream_id"), "stream_id")
    frame_id = _integer(payload.get("frame_id"), "frame_id")
    capture_ns = _integer(payload.get("capture_timestamp_ns"), "capture_timestamp_ns")
    action_id = _integer(payload.get("action_id"), "action_id")
    _require(update.get("stream_id") == stream_id, "edge/update stream identity mismatch")
    _require(_integer(update.get("frame_id"), "update.frame_id") == frame_id,
             "edge/update frame identity mismatch")
    _require(
        _integer(update.get("capture_timestamp_ns"), "update.capture_timestamp_ns")
        == capture_ns,
        "edge/update capture identity mismatch",
    )
    _require(_integer(update.get("action_id"), "update.action_id") == action_id,
             "edge/update action identity mismatch")
    records = update.get("records")
    _require(isinstance(records, list), "object-update records must be a list")
    for record in records:
        _require(isinstance(record, Mapping), "object-update record must be an object")
        _require(record.get("stream_id") == stream_id, "record stream identity mismatch")
        _require(_integer(record.get("frame_id"), "record.frame_id") == frame_id,
                 "record frame identity mismatch")
        _require(
            _integer(record.get("capture_timestamp_ns"), "record.capture_timestamp_ns")
            == capture_ns,
            "record capture identity mismatch",
        )
    return _batch(
        ue_id=ue_id,
        stream_id=stream_id,
        session_id=session_id,
        clock_domain=clock_domain,
        frame_id=frame_id,
        capture_timestamp_ns=capture_ns,
        received_clock_domain=clock_domain,
        received_timestamp_ns=received_timestamp_ns,
        action_id=action_id,
        source_schema=SPLITFUSION_OBJECT_UPDATE_SCHEMA,
        source_payload=payload,
        records=records,
    )


def from_legacy_spatial_packet(
    payload: Mapping[str, Any],
    *,
    ue_id: str,
    session_id: str,
    received_timestamp_ns: int,
) -> ObservationBatch:
    """Adapt the older two-ego CARLA packet without inventing wall-clock AoI."""

    _require(
        payload.get("schema") == LEGACY_SPATIAL_PACKET_SCHEMA,
        "unexpected legacy spatial-packet schema",
    )
    records = payload.get("objects")
    _require(isinstance(records, list), "legacy objects must be a list")
    for record in records:
        _require(isinstance(record, Mapping), "legacy object must be an object")
    carla_timestamp_s = _finite(payload.get("carla_timestamp"), "carla_timestamp")
    _require(carla_timestamp_s >= 0.0, "carla_timestamp must be nonnegative")
    return _batch(
        ue_id=ue_id,
        stream_id=payload.get("stream_id"),
        session_id=session_id,
        clock_domain="carla_simulation_ns",
        frame_id=payload.get("frame_id"),
        capture_timestamp_ns=round(carla_timestamp_s * 1_000_000_000),
        received_clock_domain="unix_wall_ns",
        received_timestamp_ns=received_timestamp_ns,
        action_id=None,
        source_schema=LEGACY_SPATIAL_PACKET_SCHEMA,
        source_payload=payload,
        records=records,
    )

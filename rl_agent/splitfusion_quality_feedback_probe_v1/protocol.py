"""Compact wire contract for privileged exact-quality progress feedback.

The packet is deliberately *not* the detailed scientific record. The full,
self-describing score/timing row stays on the edge and is hash-bound by ``dh``.
Only primitive policy inputs cross the OAI downlink in one sub-MTU datagram.
The positional arrays and their names are versioned together below.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_direct_edge_map_v1 import protocol as direct_protocol


QUALITY_EVALUATED_ACK_SCHEMA = "sf_priv_quality_ack.v1"
QUALITY_EVALUATION_FAILED_ACK_SCHEMA = "sf_priv_quality_fail.v1"
PROTOCOL_VERSION = 1
SOURCE = "EDGE_CARLA_GT"
MAX_WIRE_BYTES = 1200

IDENTITY_FIELDS = (
    "run_id", "cell_id", "stream_id", "frame_id", "action_id", "profile_id",
    "capture_timestamp_ns",
)
REQUIRED_IDENTITY = IDENTITY_FIELDS
TIMING_FIELDS = (
    "model_ready_wall_ns",
    "final_prediction_ready_wall_ns",
    "gt_ready_wall_ns",
    "evaluation_enqueued_wall_ns",
    "evaluation_started_wall_ns",
    "evaluation_completed_wall_ns",
    "ack_emit_start_wall_ns",
)
QUALITY_FIELDS = (
    "seg_vehicle_iou", "seg_person_iou", "seg_miou_3class",
    "vehicle_recall", "vehicle_xy_error_m", "vehicle_footprint_iou",
    "person_recall", "person_xy_error_m", "person_footprint_iou",
    "gt_vehicle_pixels", "gt_person_pixels",
    "vehicle_tp", "vehicle_fn", "person_tp", "person_fn",
)


class QualityProtocolError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualityProtocolError(message)


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def detail_digest(document: Mapping[str, Any]) -> str:
    return digest(document)


def _finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite scalar at {path}")
        return
    raise QualityProtocolError(f"unsupported value at {path}: {type(value).__name__}")


def identity_dict(document: Mapping[str, Any]) -> dict[str, Any]:
    values = document.get("id")
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        and len(values) == len(IDENTITY_FIELDS),
        "quality ACK identity layout drift",
    )
    return dict(zip(IDENTITY_FIELDS, values))


def identity(document: Mapping[str, Any]) -> tuple[str, str, str, int, int, str, int]:
    values = identity_dict(document)
    return (
        str(values["run_id"]), str(values["cell_id"]), str(values["stream_id"]),
        int(values["frame_id"]), int(values["action_id"]),
        str(values["profile_id"]), int(values["capture_timestamp_ns"]),
    )


def timing_dict(document: Mapping[str, Any]) -> dict[str, int | None]:
    values = document.get("t")
    _require(isinstance(values, list), "quality ACK lacks timing array")
    _require(len(values) == len(TIMING_FIELDS), "quality ACK timing layout drift")
    return {
        name: None if value is None else int(value)
        for name, value in zip(TIMING_FIELDS, values)
    }


def quality_dict(document: Mapping[str, Any]) -> dict[str, Any]:
    values = document.get("q")
    if values in (None, []):
        return {}
    _require(isinstance(values, list), "quality ACK quality layout drift")
    _require(len(values) == len(QUALITY_FIELDS), "quality ACK quality layout drift")
    return dict(zip(QUALITY_FIELDS, values))


def _primitive_quality(quality: Mapping[str, Any] | None) -> list[Any]:
    if not quality:
        return []
    segmentation = dict(quality.get("segmentation") or {})
    localization = dict(quality.get("localization") or {})
    vehicle = dict(localization.get("vehicle") or {})
    person = dict(localization.get("person") or {})
    values = {
        "seg_vehicle_iou": segmentation.get("miou_vehicle_iou"),
        "seg_person_iou": segmentation.get("miou_person_iou"),
        "seg_miou_3class": segmentation.get("miou_3class_macro"),
        "vehicle_recall": vehicle.get("recall"),
        "vehicle_xy_error_m": vehicle.get("source_time_world_xy_error_m"),
        "vehicle_footprint_iou": vehicle.get("footprint_iou"),
        "person_recall": person.get("recall"),
        "person_xy_error_m": person.get("source_time_world_xy_error_m"),
        "person_footprint_iou": person.get("footprint_iou"),
        "gt_vehicle_pixels": segmentation.get("gt_vehicle_pixels"),
        "gt_person_pixels": segmentation.get("gt_person_pixels"),
        "vehicle_tp": vehicle.get("tp"), "vehicle_fn": vehicle.get("fn"),
        "person_tp": person.get("tp"), "person_fn": person.get("fn"),
    }
    return [values[name] for name in QUALITY_FIELDS]


def validate(document: Mapping[str, Any]) -> None:
    schema = str(document.get("s") or "")
    _require(
        schema in {QUALITY_EVALUATED_ACK_SCHEMA, QUALITY_EVALUATION_FAILED_ACK_SCHEMA},
        f"quality ACK schema drift: {schema!r}",
    )
    _require(int(document.get("v", 0)) == PROTOCOL_VERSION, "protocol version drift")
    _require(document.get("src") == SOURCE, "quality ACK source drift")
    _require(document.get("nt") is True, "quality ACK must be non-terminal")
    _require(
        document.get("pg") is True and document.get("dp") is False,
        "quality ACK must disclose privileged, non-deployable ground truth",
    )
    values = identity_dict(document)
    for name in IDENTITY_FIELDS:
        _require(values[name] not in (None, ""), f"quality ACK has empty {name}")
    _require(int(values["frame_id"]) >= 0, "frame_id must be non-negative")
    _require(0 <= int(values["action_id"]) < 72, "action_id outside catalog")
    _require(int(values["capture_timestamp_ns"]) > 0, "capture timestamp must be positive")
    _require(int(document.get("cf", -1)) == int(values["frame_id"]), "CARLA frame drift")
    timing = timing_dict(document)
    if schema == QUALITY_EVALUATED_ACK_SCHEMA:
        for name in TIMING_FIELDS:
            _require(int(timing.get(name) or 0) > 0, f"quality ACK lacks {name}")
        ordered = [
            int(timing["final_prediction_ready_wall_ns"] or 0),
            int(timing["evaluation_enqueued_wall_ns"] or 0),
            int(timing["evaluation_started_wall_ns"] or 0),
            int(timing["evaluation_completed_wall_ns"] or 0),
            int(timing["ack_emit_start_wall_ns"] or 0),
        ]
        _require(
            all(right >= left for left, right in zip(ordered, ordered[1:])),
            "quality ACK wall-clock stages are inverted",
        )
        _require(len(quality_dict(document)) == len(QUALITY_FIELDS), "scores absent")
        _require(not document.get("r"), "successful quality ACK has failure")
    else:
        _require(bool(str(document.get("r") or "")), "failed ACK lacks reason")
        _require(document.get("q") in (None, []), "failed ACK carries scores")
        _require(
            int(timing.get("ack_emit_start_wall_ns") or 0) > 0,
            "failed ACK lacks observed emit-start time",
        )
    _require(
        isinstance(document.get("dh"), str) and len(str(document["dh"])) == 64,
        "quality ACK lacks edge-detail digest",
    )
    _finite_tree(document)
    direct_protocol.assert_no_object_records(document)
    payload = canonical_bytes(document)
    _require(
        len(payload) <= MAX_WIRE_BYTES,
        f"quality ACK exceeds {MAX_WIRE_BYTES}-B wire budget: {len(payload)} B",
    )


def build_detail(
    *, identity_fields: Mapping[str, Any], frozen_carla_frame_id: int,
    timing: Mapping[str, Any], quality: Mapping[str, Any] | None,
    evaluator_mode: str, failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "schema": "splitfusion_privileged_quality_detail.v1",
        "privileged_carla_ground_truth": True,
        "deployable_feedback": False,
        "terminal": False,
        **{name: identity_fields[name] for name in IDENTITY_FIELDS},
        "capture_id": f"{identity_fields['stream_id']}:{int(identity_fields['frame_id'])}",
        "frozen_carla_frame_id": int(frozen_carla_frame_id),
        "evaluator_mode": str(evaluator_mode),
        "timing": dict(timing),
        "quality": dict(quality or {}),
        "failure_reason": str(failure_reason),
    }


def build_ack(
    *, identity_fields: Mapping[str, Any], frozen_carla_frame_id: int,
    timing: Mapping[str, Any], quality: Mapping[str, Any] | None,
    evaluator_mode: str, detail_sha256: str, failure_reason: str = "",
) -> dict[str, Any]:
    success = not failure_reason
    document: dict[str, Any] = {
        "s": QUALITY_EVALUATED_ACK_SCHEMA if success else QUALITY_EVALUATION_FAILED_ACK_SCHEMA,
        "v": PROTOCOL_VERSION, "src": SOURCE, "nt": True, "pg": True, "dp": False,
        "e": "OK" if success else "FAIL",
        "id": [identity_fields[name] for name in IDENTITY_FIELDS],
        "cf": int(frozen_carla_frame_id), "m": str(evaluator_mode),
        "t": [timing.get(name) for name in TIMING_FIELDS],
        "q": _primitive_quality(quality) if success else [],
        "dh": str(detail_sha256), "r": str(failure_reason),
    }
    validate(document)
    return document

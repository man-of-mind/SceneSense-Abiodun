"""Versioned wire contract for direct edge-to-map publication.

The corrected architecture is

    UE feature -> OAI uplink -> edge inference
      -> direct edge-to-map publication and installation
      -> compact feedback/agent-credit message to the UE

Three compact, strictly versioned documents implement it:

``splitfusion_direct_object_map_update.v1``
    Edge -> spatial map. The only message that carries object records. It is
    published on an edge-local container-to-host path and must never be
    addressed to the UE.

``splitfusion_direct_map_feedback.v1``
    Spatial map -> UE. Emitted strictly after the map inserted (or refused) the
    update under its authoritative state lock. Carries identity, install
    timestamp, outcome, agent credit and map age at installation. It carries no
    object records and no dense semantic mask.

``splitfusion_edge_terminal_control.v1``
    Edge -> UE. The terminal for a transmission obligation the edge ended
    before publication (stale or superseded), so a frame the edge deliberately
    replaced with fresher work is never mistaken for a network failure. It also
    carries no object records.

Exactly one terminal outcome exists per UE transmission obligation: either a
map feedback terminal, an edge terminal control, or the UE-local feedback
timeout. The three sources are disjoint by construction -- the edge emits a
terminal control only for frames it did not publish, and the map emits a
terminal only for frames the edge published.
"""

from __future__ import annotations

import ipaddress
import json
import math
from typing import Any, Mapping, Sequence


DIRECT_OBJECT_MAP_UPDATE_SCHEMA = "splitfusion_direct_object_map_update.v1"
DIRECT_MAP_FEEDBACK_SCHEMA = "splitfusion_direct_map_feedback.v1"
EDGE_TERMINAL_CONTROL_SCHEMA = "splitfusion_edge_terminal_control.v1"

PROTOCOL_VERSION = 1

# Registered scheduler outcomes. Every UE transmission obligation ends in
# exactly one of these.
OUTCOME_RESULT_INSTALLED = "RESULT_INSTALLED"
OUTCOME_SUPERSEDED_PENDING = "SUPERSEDED_PENDING"
OUTCOME_STALE_BEFORE_EDGE = "STALE_BEFORE_EDGE"
OUTCOME_STALE_BEFORE_MAP = "STALE_BEFORE_MAP"
OUTCOME_TRANSPORT_INCOMPLETE = "TRANSPORT_INCOMPLETE"
OUTCOME_MAP_REJECTED = "MAP_REJECTED"
OUTCOME_FEEDBACK_TIMEOUT = "FEEDBACK_TIMEOUT"

TERMINAL_OUTCOMES = (
    OUTCOME_RESULT_INSTALLED,
    OUTCOME_SUPERSEDED_PENDING,
    OUTCOME_STALE_BEFORE_EDGE,
    OUTCOME_STALE_BEFORE_MAP,
    OUTCOME_TRANSPORT_INCOMPLETE,
    OUTCOME_MAP_REJECTED,
    OUTCOME_FEEDBACK_TIMEOUT,
)

# Outcomes a map feedback message may carry.
MAP_OUTCOMES = (
    OUTCOME_RESULT_INSTALLED,
    OUTCOME_SUPERSEDED_PENDING,
    OUTCOME_STALE_BEFORE_MAP,
    OUTCOME_MAP_REJECTED,
)
# Outcomes an edge terminal control may carry.
EDGE_CONTROL_OUTCOMES = (
    OUTCOME_STALE_BEFORE_EDGE,
    OUTCOME_SUPERSEDED_PENDING,
    OUTCOME_STALE_BEFORE_MAP,
    OUTCOME_TRANSPORT_INCOMPLETE,
)

# Agent-credit classification. A superseded frame is credited as deliberately
# replaced work, never as a network failure.
CREDIT_INSTALLED_ON_TIME = "CREDIT_INSTALLED_ON_TIME"
CREDIT_INSTALLED_LATE = "CREDIT_INSTALLED_LATE"
CREDIT_SUPERSEDED_BY_FRESHER = "CREDIT_SUPERSEDED_BY_FRESHER"
CREDIT_STALE_AT_SOURCE = "CREDIT_STALE_AT_SOURCE"
CREDIT_STALE_AT_MAP = "CREDIT_STALE_AT_MAP"
CREDIT_NETWORK_INCOMPLETE = "CREDIT_NETWORK_INCOMPLETE"
CREDIT_REJECTED = "CREDIT_REJECTED"
CREDIT_NO_FEEDBACK = "CREDIT_NO_FEEDBACK"

AGENT_CREDITS = (
    CREDIT_INSTALLED_ON_TIME,
    CREDIT_INSTALLED_LATE,
    CREDIT_SUPERSEDED_BY_FRESHER,
    CREDIT_STALE_AT_SOURCE,
    CREDIT_STALE_AT_MAP,
    CREDIT_NETWORK_INCOMPLETE,
    CREDIT_REJECTED,
    CREDIT_NO_FEEDBACK,
)

# Fields the map update must carry for the map to validate identity.
REQUIRED_UPDATE_IDENTITY = (
    "run_id",
    "cell_id",
    "stream_id",
    "frame_id",
    "action_id",
    "profile_id",
    "capture_timestamp_ns",
)

# Keys that must never appear anywhere in a message sent to the UE.
FORBIDDEN_UE_KEYS = (
    "records",
    "objects",
    "semantic_labels_b64",
    "semantic_labels",
    "label_map",
    "dense_mask",
)


class DirectMapProtocolError(RuntimeError):
    """A direct edge/map message violated the versioned contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectMapProtocolError(message)


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def assert_finite_tree(value: Any, *, path: str = "$") -> int:
    """Return the number of scalars checked, raising on any non-finite float."""

    if isinstance(value, Mapping):
        total = 0
        for key, item in value.items():
            total += assert_finite_tree(item, path=f"{path}.{key}")
        return total
    if isinstance(value, (list, tuple)):
        total = 0
        for index, item in enumerate(value):
            total += assert_finite_tree(item, path=f"{path}[{index}]")
        return total
    if value is None or isinstance(value, str):
        return 1
    _require(_finite(value), f"non-finite object record value at {path}: {value!r}")
    return 1


def encode(document: Mapping[str, Any]) -> bytes:
    """Canonical, NaN-refusing encoding used on every direct/control path."""

    return json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def decode(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectMapProtocolError(f"undecodable datagram: {exc}") from exc
    _require(isinstance(value, dict), "datagram payload is not a JSON object")
    return value


def is_ue_address(host: str, ue_hosts: Sequence[str]) -> bool:
    """True when ``host`` is one of the UE tunnel addresses."""

    try:
        candidate = ipaddress.ip_address(str(host))
    except ValueError:
        return str(host) in {str(value) for value in ue_hosts}
    for value in ue_hosts:
        try:
            if candidate == ipaddress.ip_address(str(value)):
                return True
        except ValueError:
            continue
    return False


def assert_direct_map_endpoint(
    host: str, port: int, *, ue_hosts: Sequence[str], forbidden_ports: Sequence[int] = ()
) -> None:
    """Refuse any object-map endpoint that would re-create the UE detour.

    This is the runtime half of the static address audit: the direct publisher
    refuses to construct a socket aimed at the UE tunnel, at loopback (which the
    edge container cannot reach and which would silently disable the map), or at
    a port reserved for the UE result path.
    """

    _require(bool(str(host).strip()), "direct map host is empty")
    _require(int(port) > 0, "direct map port is invalid")
    _require(
        not is_ue_address(host, ue_hosts),
        f"direct object-map endpoint must not target the UE tunnel address {host}",
    )
    _require(
        int(port) not in {int(value) for value in forbidden_ports},
        f"direct object-map endpoint must not use UE result port {port}",
    )
    try:
        address = ipaddress.ip_address(str(host))
    except ValueError:
        return
    _require(
        not address.is_loopback,
        "direct object-map endpoint must not be loopback: the edge container "
        "cannot reach the host loopback interface",
    )
    _require(
        not address.is_unspecified,
        "direct object-map endpoint must be an explicit address, not 0.0.0.0",
    )


def assert_no_object_records(document: Mapping[str, Any]) -> None:
    """Prove a UE-bound message is an ACK/control message, not a map update."""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _require(
                    str(key) not in FORBIDDEN_UE_KEYS,
                    f"UE-bound message carries forbidden key {path}.{key}",
                )
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(document, "$")
    _require(
        str(document.get("schema") or "") != DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
        "the object map update itself must never be sent to the UE",
    )


def build_object_map_update(
    *,
    run_id: str,
    cell_id: str,
    stream_id: str,
    frame_id: int,
    sequence_id: int,
    action_id: int,
    profile_id: str,
    decoder_identity: str,
    capture_timestamp_ns: int,
    carla_timestamp: float,
    records: Sequence[Mapping[str, Any]],
    service_deadline_at: float,
    ack_timeout_at: float,
    edge_timing: Mapping[str, Any],
    segmentation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the one message that carries object records, edge -> map."""

    record_list = [dict(item) for item in records]
    scalars = assert_finite_tree(record_list, path="$.records")
    document = {
        "schema": DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": str(run_id),
        "cell_id": str(cell_id),
        "stream_id": str(stream_id),
        "frame_id": int(frame_id),
        "sequence_id": int(sequence_id),
        "action_id": int(action_id),
        "profile_id": str(profile_id),
        "decoder_identity": str(decoder_identity),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "capture_timestamp": int(capture_timestamp_ns) / 1_000_000_000.0,
        "carla_timestamp": float(carla_timestamp),
        "service_deadline_at": float(service_deadline_at),
        "ack_timeout_at": float(ack_timeout_at),
        "record_count": len(record_list),
        "record_scalar_count": int(scalars),
        "records": record_list,
        "edge_timing": dict(edge_timing),
        "segmentation": dict(segmentation),
    }
    validate_object_map_update(document)
    return document


def validate_object_map_update(document: Mapping[str, Any]) -> None:
    """Map-side validation of run/cell/stream/frame/action identity and finiteness."""

    _require(
        document.get("schema") == DIRECT_OBJECT_MAP_UPDATE_SCHEMA,
        f"object map update schema drift: {document.get('schema')!r}",
    )
    _require(
        int(document.get("protocol_version", 0)) == PROTOCOL_VERSION,
        "object map update protocol version drift",
    )
    for field in REQUIRED_UPDATE_IDENTITY:
        _require(field in document, f"object map update lacks {field}")
        _require(
            document[field] not in (None, ""),
            f"object map update has an empty {field}",
        )
    _require(int(document["frame_id"]) >= 0, "frame_id must be non-negative")
    _require(0 <= int(document["action_id"]) < 72, "action_id is outside the catalog")
    _require(
        int(document["capture_timestamp_ns"]) > 0,
        "capture_timestamp_ns must be positive",
    )
    records = document.get("records")
    _require(isinstance(records, list), "object map update records must be a list")
    _require(
        int(document.get("record_count", -1)) == len(records),
        "object map update record_count disagrees with records",
    )
    assert_finite_tree(records, path="$.records")
    _require(
        float(document["ack_timeout_at"]) > float(document["service_deadline_at"]),
        "ACK timeout must exceed the service deadline",
    )


def update_identity(document: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """The duplicate/supersession key: run, cell, stream, frame."""

    return (
        str(document["run_id"]),
        str(document["cell_id"]),
        str(document["stream_id"]),
        int(document["frame_id"]),
    )


def classify_agent_credit(
    outcome: str,
    *,
    install_timestamp: float | None = None,
    service_deadline_at: float | None = None,
) -> str:
    """Map a terminal outcome to its agent-credit class.

    A frame the edge or map deliberately replaced with fresher work is credited
    ``CREDIT_SUPERSEDED_BY_FRESHER``; it is never credited as a network failure.
    """

    _require(outcome in TERMINAL_OUTCOMES, f"unregistered outcome: {outcome!r}")
    if outcome == OUTCOME_RESULT_INSTALLED:
        _require(
            install_timestamp is not None and service_deadline_at is not None,
            "installed credit requires an install timestamp and service deadline",
        )
        return (
            CREDIT_INSTALLED_ON_TIME
            if float(install_timestamp) <= float(service_deadline_at)
            else CREDIT_INSTALLED_LATE
        )
    return {
        OUTCOME_SUPERSEDED_PENDING: CREDIT_SUPERSEDED_BY_FRESHER,
        OUTCOME_STALE_BEFORE_EDGE: CREDIT_STALE_AT_SOURCE,
        OUTCOME_STALE_BEFORE_MAP: CREDIT_STALE_AT_MAP,
        OUTCOME_TRANSPORT_INCOMPLETE: CREDIT_NETWORK_INCOMPLETE,
        OUTCOME_MAP_REJECTED: CREDIT_REJECTED,
        OUTCOME_FEEDBACK_TIMEOUT: CREDIT_NO_FEEDBACK,
    }[outcome]


def build_map_feedback(
    *,
    update: Mapping[str, Any],
    outcome: str,
    terminal: bool,
    install_timestamp: float | None,
    map_ingest_at: float,
    feedback_emit_at: float,
    map_age_at_install_ms: float | None,
    rejection_reason: str = "",
    direct_update_bytes: int = 0,
    direct_update_datagrams: int = 0,
    superseded_by_frame_id: int | None = None,
) -> dict[str, Any]:
    """Assemble the compact map -> UE ACK/agent-credit/control message.

    It carries identity, the install timestamp, the outcome, agent credit and
    the map age at installation. It never carries object records.
    """

    _require(outcome in MAP_OUTCOMES, f"invalid map feedback outcome: {outcome!r}")
    if outcome == OUTCOME_RESULT_INSTALLED:
        _require(
            install_timestamp is not None,
            "RESULT_INSTALLED feedback requires an install timestamp",
        )
    else:
        _require(
            install_timestamp is None,
            "only RESULT_INSTALLED feedback may carry an install timestamp",
        )
    credit = classify_agent_credit(
        outcome,
        install_timestamp=install_timestamp,
        service_deadline_at=float(update["service_deadline_at"]),
    )
    document = {
        "schema": DIRECT_MAP_FEEDBACK_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "source": "EDGE_SPATIAL_MAP",
        "run_id": str(update["run_id"]),
        "cell_id": str(update["cell_id"]),
        "stream_id": str(update["stream_id"]),
        "frame_id": int(update["frame_id"]),
        "capture_id": f"{update['stream_id']}:{int(update['frame_id'])}",
        "action_id": int(update["action_id"]),
        "profile_id": str(update["profile_id"]),
        "capture_timestamp_ns": int(update["capture_timestamp_ns"]),
        "capture_timestamp": int(update["capture_timestamp_ns"]) / 1_000_000_000.0,
        "service_deadline_at": float(update["service_deadline_at"]),
        "ack_timeout_at": float(update["ack_timeout_at"]),
        "outcome": str(outcome),
        "agent_credit": credit,
        "terminal": bool(terminal),
        "accepted": outcome == OUTCOME_RESULT_INSTALLED,
        "install_timestamp": ("" if install_timestamp is None else float(install_timestamp)),
        "map_ingest_at": float(map_ingest_at),
        "feedback_emit_at": float(feedback_emit_at),
        "map_age_at_install_ms": (
            "" if map_age_at_install_ms is None else float(map_age_at_install_ms)
        ),
        "rejection_reason": str(rejection_reason),
        "direct_update_bytes": int(direct_update_bytes),
        "direct_update_datagrams": int(direct_update_datagrams),
        "superseded_by_frame_id": (
            "" if superseded_by_frame_id is None else int(superseded_by_frame_id)
        ),
        "edge_timing": dict(update.get("edge_timing") or {}),
    }
    assert_no_object_records(document)
    validate_map_feedback(document)
    return document


def validate_map_feedback(document: Mapping[str, Any]) -> None:
    _require(
        document.get("schema") == DIRECT_MAP_FEEDBACK_SCHEMA,
        f"map feedback schema drift: {document.get('schema')!r}",
    )
    _require(
        int(document.get("protocol_version", 0)) == PROTOCOL_VERSION,
        "map feedback protocol version drift",
    )
    _require(document.get("outcome") in MAP_OUTCOMES, "map feedback outcome is invalid")
    _require(
        document.get("agent_credit") in AGENT_CREDITS,
        "map feedback agent credit is invalid",
    )
    for field in ("run_id", "cell_id", "stream_id", "frame_id", "action_id", "profile_id"):
        _require(field in document, f"map feedback lacks {field}")
    if document["outcome"] == OUTCOME_RESULT_INSTALLED:
        _require(
            document.get("install_timestamp") not in (None, ""),
            "RESULT_INSTALLED feedback lacks install_timestamp",
        )
        _require(bool(document.get("accepted")), "RESULT_INSTALLED must be accepted")
    assert_no_object_records(document)


def build_edge_terminal_control(
    *,
    run_id: str,
    cell_id: str,
    stream_id: str,
    frame_id: int,
    action_id: int,
    profile_id: str,
    capture_timestamp_ns: int,
    service_deadline_at: float,
    ack_timeout_at: float,
    outcome: str,
    stage: str,
    age_ms: float,
    emit_at: float,
    superseded_by_frame_id: int | None = None,
) -> dict[str, Any]:
    """Assemble the compact edge -> UE terminal for an unpublished frame.

    The selected action identity is preserved so a deliberately replaced frame
    is attributed to the edge/map scheduler, not to the radio.
    """

    _require(
        outcome in EDGE_CONTROL_OUTCOMES,
        f"invalid edge terminal outcome: {outcome!r}",
    )
    document = {
        "schema": EDGE_TERMINAL_CONTROL_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "source": "EDGE_INFERENCE_SERVICE",
        "run_id": str(run_id),
        "cell_id": str(cell_id),
        "stream_id": str(stream_id),
        "frame_id": int(frame_id),
        "capture_id": f"{stream_id}:{int(frame_id)}",
        "action_id": int(action_id),
        "profile_id": str(profile_id),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "capture_timestamp": int(capture_timestamp_ns) / 1_000_000_000.0,
        "service_deadline_at": float(service_deadline_at),
        "ack_timeout_at": float(ack_timeout_at),
        "outcome": str(outcome),
        "agent_credit": classify_agent_credit(outcome),
        "terminal": True,
        "accepted": False,
        "stage": str(stage),
        "age_ms": float(age_ms),
        "emit_at": float(emit_at),
        "superseded_by_frame_id": (
            "" if superseded_by_frame_id is None else int(superseded_by_frame_id)
        ),
    }
    assert_no_object_records(document)
    validate_edge_terminal_control(document)
    return document


def validate_edge_terminal_control(document: Mapping[str, Any]) -> None:
    _require(
        document.get("schema") == EDGE_TERMINAL_CONTROL_SCHEMA,
        f"edge terminal control schema drift: {document.get('schema')!r}",
    )
    _require(
        int(document.get("protocol_version", 0)) == PROTOCOL_VERSION,
        "edge terminal control protocol version drift",
    )
    _require(
        document.get("outcome") in EDGE_CONTROL_OUTCOMES,
        "edge terminal control outcome is invalid",
    )
    _require(
        document.get("agent_credit") in AGENT_CREDITS,
        "edge terminal control agent credit is invalid",
    )
    _require(bool(document.get("terminal")), "edge terminal control must be terminal")
    assert_no_object_records(document)

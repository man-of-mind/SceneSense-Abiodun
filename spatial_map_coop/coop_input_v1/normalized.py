"""Transport-independent normalized cooperative-map input.

One type, two adapters, no guessing. Everything an adapter cannot prove from its
source document is ``None`` and is reported as ``N/A`` downstream; nothing is
defaulted to zero, inferred from a name, or translated between clock domains.

Clock domains are carried as data, never assumed. ``unix_wall_ns`` and
``carla_simulation_ns`` are different physical clocks with no measured offset
contract in this branch, so any quantity that would subtract one from the other
is refused rather than computed (see ``clocks_compatible``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


NORMALIZED_SCHEMA = "scenesense.normalized_coop_map_input.v1"

# Registered transports. The label states what actually carried the bytes; it is
# never a performance claim.
TRANSPORT_HTTP_DEMO = "HTTP_JSON_POST_LOCALHOST_DEMO"
TRANSPORT_DIRECT_UDP = "UDP_ZLIB_CHUNKED_EDGE_LOCAL_DIRECT_V1"

# Registered clock domains.
CLOCK_UNIX_WALL_NS = "unix_wall_ns"
CLOCK_UNIX_WALL_S = "unix_wall_s"
CLOCK_CARLA_SIM_NS = "carla_simulation_ns"

# Pose availability states. The viewer must render the negative states as an
# explicit "pose unavailable" badge and draw no FoV wedge.
POSE_AVAILABLE = "POSE_AVAILABLE"
POSE_UNAVAILABLE_NOT_IN_PROTOCOL = "POSE_UNAVAILABLE_NOT_CARRIED_BY_PROTOCOL_V1"
POSE_UNAVAILABLE_NO_SIDECAR = "POSE_UNAVAILABLE_NO_MATCHING_SIDECAR"

POSE_STATES = (
    POSE_AVAILABLE,
    POSE_UNAVAILABLE_NOT_IN_PROTOCOL,
    POSE_UNAVAILABLE_NO_SIDECAR,
)


class CoopInputError(ValueError):
    """A cooperative-map input document violated the normalization contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoopInputError(message)


def nonempty_str(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    result = value.strip()
    _require(bool(result), f"{label} must not be empty")
    return result


def strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer, not a bool")
    _require(isinstance(value, int), f"{label} must be an integer")
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{label} must be >= {minimum}")
    return result


def strict_finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be a number, not a bool")
    _require(isinstance(value, (int, float)), f"{label} must be a number")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def optional_finite(value: Any) -> float | None:
    """A finite float, or None for absent/blank/non-finite. Never zero-filled."""

    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def canonical_json(value: Any, label: str = "document") -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CoopInputError(f"{label} is not canonical JSON: {exc}") from exc


def sha256_of(value: Any, label: str = "document") -> str:
    return hashlib.sha256(canonical_json(value, label).encode("utf-8")).hexdigest()


def assert_finite_records(records: Sequence[Mapping[str, Any]]) -> int:
    """Count record scalars, refusing any non-finite float.

    Mirrors the intent of the pinned ``protocol.assert_finite_tree`` for the
    records subtree only. It does not replace the production validator; see
    ``reference_pins``.
    """

    def walk(value: Any, path: str) -> int:
        if isinstance(value, Mapping):
            return sum(walk(item, f"{path}.{key}") for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return sum(walk(item, f"{path}[{i}]") for i, item in enumerate(value))
        if value is None or isinstance(value, str) or isinstance(value, bool):
            return 1
        _require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"non-finite record value at {path}: {value!r}",
        )
        return 1

    return sum(walk(record, f"$.records[{i}]") for i, record in enumerate(records))


def freeze_records(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Retain p025 records verbatim behind read-only views.

    No key is renamed, dropped, reordered in meaning, rounded or recomputed. The
    normalization seam is explicitly *not* allowed to touch perception output.
    """

    frozen: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"records[{index}] must be an object")
        frozen.append(MappingProxyType(dict(record)))
    return tuple(frozen)


@dataclass(frozen=True)
class ClockStamp:
    """A timestamp that always travels with the name of its clock."""

    timestamp_ns: int
    clock_domain: str

    def __post_init__(self) -> None:
        strict_int(self.timestamp_ns, "timestamp_ns")
        nonempty_str(self.clock_domain, "clock_domain")

    @property
    def seconds(self) -> float:
        return self.timestamp_ns / 1_000_000_000.0

    def as_dict(self) -> dict[str, Any]:
        return {"timestamp_ns": self.timestamp_ns, "clock_domain": self.clock_domain}


def clocks_compatible(first: ClockStamp | None, second: ClockStamp | None) -> bool:
    """True only when two stamps are on the same registered clock.

    There is no measured offset contract between ``carla_simulation_ns`` and
    ``unix_wall_ns`` in this branch, so mixing them is refused rather than
    silently translated.
    """

    if first is None or second is None:
        return False
    return first.clock_domain == second.clock_domain


@dataclass(frozen=True)
class EdgeTimingProvenance:
    """Edge stage stamps exactly as the source carried them.

    ``fields`` holds only entries that were present and finite. An absent stage
    is absent, not zero.
    """

    source: str
    clock_domain: str | None
    fields: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def absent(cls, source: str = "ABSENT") -> "EdgeTimingProvenance":
        return cls(source=source, clock_domain=None, fields=MappingProxyType({}))

    @classmethod
    def from_mapping(
        cls, value: Any, *, source: str, clock_domain: str
    ) -> "EdgeTimingProvenance":
        if not isinstance(value, Mapping):
            return cls.absent(source="ABSENT_MALFORMED")
        kept: dict[str, float] = {}
        for key, item in value.items():
            numeric = optional_finite(item)
            if numeric is not None:
                kept[str(key)] = numeric
        if not kept:
            return cls.absent(source="ABSENT_EMPTY")
        return cls(
            source=source,
            clock_domain=clock_domain,
            fields=MappingProxyType(dict(sorted(kept.items()))),
        )

    def get(self, name: str) -> float | None:
        return self.fields.get(name)

    @property
    def available(self) -> bool:
        return bool(self.fields)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "clock_domain": self.clock_domain,
            "fields": dict(self.fields),
        }


@dataclass(frozen=True)
class MapTerminal:
    """The map-side terminal outcome, when the source proved one."""

    source: str
    outcome: str | None = None
    agent_credit: str | None = None
    terminal: bool | None = None
    accepted: bool | None = None
    install_timestamp: ClockStamp | None = None
    map_age_at_install_ms: float | None = None
    rejection_reason: str = ""
    superseded_by_frame_id: int | None = None

    @classmethod
    def absent(cls, source: str = "ABSENT_NO_TERMINAL_EVIDENCE") -> "MapTerminal":
        return cls(source=source)

    @property
    def available(self) -> bool:
        return self.outcome is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "outcome": self.outcome,
            "agent_credit": self.agent_credit,
            "terminal": self.terminal,
            "accepted": self.accepted,
            "install_timestamp": (
                None if self.install_timestamp is None else self.install_timestamp.as_dict()
            ),
            "map_age_at_install_ms": self.map_age_at_install_ms,
            "rejection_reason": self.rejection_reason,
            "superseded_by_frame_id": self.superseded_by_frame_id,
        }


@dataclass(frozen=True)
class NormalizedPose:
    """A measured ego/sensor pose plus the identity it was bound to.

    Only ever constructed from a source that actually carried a pose. There is no
    constructor that fabricates one, and no CARLA actor is consulted: querying
    ground-truth actors would put truth behind the perception path.
    """

    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float
    horizontal_fov_deg: float
    source: str
    binding: tuple[str, str, str, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "pitch": self.pitch, "yaw": self.yaw, "roll": self.roll,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "source": self.source,
            "binding": list(self.binding),
        }


@dataclass(frozen=True)
class NormalizedCoopBatch:
    """One frame of cooperative-map input, independent of how it arrived."""

    transport: str
    source_schema: str
    ue_id: str
    session_id: str
    stream_id: str
    frame_id: int
    capture: ClockStamp
    records: tuple[Mapping[str, Any], ...]
    payload_sha256: str
    records_sha256: str
    identity_binding_source: str
    protocol_version: int | None = None
    run_id: str | None = None
    cell_id: str | None = None
    sequence_id: int | None = None
    action_id: int | None = None
    profile_id: str | None = None
    decoder_identity: str | None = None
    receipt: ClockStamp | None = None
    edge_timing: EdgeTimingProvenance = field(default_factory=EdgeTimingProvenance.absent)
    map_terminal: MapTerminal = field(default_factory=MapTerminal.absent)
    pose: NormalizedPose | None = None
    pose_state: str = POSE_UNAVAILABLE_NOT_IN_PROTOCOL
    record_scalar_count: int = 0
    normalization_schema: str = NORMALIZED_SCHEMA

    def __post_init__(self) -> None:
        _require(self.pose_state in POSE_STATES, f"unregistered pose state {self.pose_state!r}")
        _require(
            (self.pose is None) == (self.pose_state != POSE_AVAILABLE),
            "pose_state must agree with the presence of a pose",
        )

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def source_key(self) -> tuple[str, str, str]:
        return self.ue_id, self.session_id, self.stream_id

    @property
    def identity(self) -> tuple[str | None, str | None, str, int]:
        """The production supersession key, with run/cell left None when absent."""

        return self.run_id, self.cell_id, self.stream_id, self.frame_id

    def with_pose(self, pose: NormalizedPose | None, state: str) -> "NormalizedCoopBatch":
        """Return a copy carrying a pose. The scientific batch is never mutated."""

        _require(state in POSE_STATES, f"unregistered pose state {state!r}")
        replacement = dict(self.__dict__)
        replacement["pose"] = pose
        replacement["pose_state"] = state
        return NormalizedCoopBatch(**replacement)

    def as_dict(self) -> dict[str, Any]:
        """Diagnostic projection. Records are summarised by hash, not re-emitted."""

        return {
            "normalization_schema": self.normalization_schema,
            "transport": self.transport,
            "source_schema": self.source_schema,
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "ue_id": self.ue_id,
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "frame_id": self.frame_id,
            "sequence_id": self.sequence_id,
            "action_id": self.action_id,
            "profile_id": self.profile_id,
            "decoder_identity": self.decoder_identity,
            "capture": self.capture.as_dict(),
            "receipt": None if self.receipt is None else self.receipt.as_dict(),
            "capture_receipt_clocks_compatible": clocks_compatible(self.capture, self.receipt),
            "record_count": self.record_count,
            "record_scalar_count": self.record_scalar_count,
            "records_sha256": self.records_sha256,
            "payload_sha256": self.payload_sha256,
            "identity_binding_source": self.identity_binding_source,
            "edge_timing": self.edge_timing.as_dict(),
            "map_terminal": self.map_terminal.as_dict(),
            "pose_state": self.pose_state,
            "pose": None if self.pose is None else self.pose.as_dict(),
        }

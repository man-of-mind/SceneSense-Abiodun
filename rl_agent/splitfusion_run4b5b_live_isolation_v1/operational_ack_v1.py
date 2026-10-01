"""Exact, GT-free operational acknowledgement for frozen Run-4B/Run-5B.

The edge emits an acknowledgement as soon as the model tail has produced the
output for an exact frame/action identity.  Spatial-map publication and CARLA
evaluation are deliberately not part of this message.  The UE measures the
only policy latency that matters here on one clock: action-open to ACK receipt
on ``CLOCK_MONOTONIC_RAW``.

The 170-ms boundary is inclusive.  An ACK received at exactly 170 ms closes the
ticket successfully; one received one nanosecond later is a late orphan and the
ticket remains a timeout.  Byte-identical duplicates are ignored, conflicting
duplicates fail closed, and a mismatched frame/action cannot close a ticket
sharing the same logical decision identity.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import re
import struct
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


ACK_DEADLINE_NS = 170_000_000
FIRST_LATE_TICK_NS = ACK_DEADLINE_NS + 1
IDENTITY_SCHEMA = "scenesense.splitfusion.run4b5b.frame_action_identity.v1"
ACK_SCHEMA = "scenesense.splitfusion.run4b5b.tail_output_ack.v1"
ACK_WIRE_SCHEMA = "scenesense.splitfusion.run4b5b.tail_output_ack_wire.v1"
ACK_STATUS = "TAIL_OUTPUT_SUCCESS"
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
OPEN_RECORD_SCHEMA = "scenesense.splitfusion.run4b5b.operational_open.v1"
OUTCOME_RECORD_SCHEMA = "scenesense.splitfusion.run4b5b.operational_outcome.v1"
ORPHAN_RECORD_SCHEMA = "scenesense.splitfusion.run4b5b.operational_orphan.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}")
_MAGIC = b"R45A"
_HEADER = struct.Struct("!4sI")
_DIGEST_BYTES = 32


class OperationalAckError(RuntimeError):
    """The operational ACK contract was violated."""


class IdentityError(OperationalAckError):
    """A frame/action identity is incomplete, unsafe, or inconsistent."""


class AckWireError(OperationalAckError):
    """An ACK packet is corrupt, foreign, or non-canonical."""


class AckIdentityConflict(OperationalAckError):
    """An ACK claims an existing decision with a different exact identity."""


class ConflictingAckError(OperationalAckError):
    """Two non-identical ACKs claim the same exact ticket."""


class OperationalEvidenceError(OperationalAckError):
    """The durable UE operational ledger is incomplete or inconsistent."""


class OperationalCreateOnlyError(OperationalEvidenceError):
    """A create-only operational evidence path already exists."""


def _require(condition: bool, message: str,
             error: type[OperationalAckError] = OperationalAckError) -> None:
    if not condition:
        raise error(message)


def _safe_id(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SAFE_ID_RE.fullmatch(value)),
             f"{field} is empty or unsafe", IdentityError)
    _require(".." not in value and "/" not in value and "\\" not in value,
             f"path syntax is forbidden in {field}", IdentityError)
    return value


def _digest(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SHA256_RE.fullmatch(value)),
             f"{field} is not a lowercase SHA-256", IdentityError)
    return value


def _nonnegative(value: Any, field: str) -> int:
    _require(type(value) is int and value >= 0,
             f"{field} must be a nonnegative integer", IdentityError)
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise OperationalAckError("value is not canonically serializable") from exc


@dataclass(frozen=True, slots=True)
class FrameActionIdentityV1:
    """Exact identity carried by the tensor, ACK, map item, and evidence row.

    ``anchor_action_id`` and ``profile_id`` are both null for a continuous
    off-anchor action.  ``mode_id``, ``q_e4``, ``keep_count`` and the execution
    bundle remain present, so an off-anchor action is never approximated by an
    anchor identity.
    """

    run_id: str
    cell_id: str
    stream_id: str
    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    mode_id: int
    q_e4: int
    keep_count: int
    anchor_action_id: Optional[int]
    profile_id: Optional[str]
    execution_bundle_sha256: str

    def __post_init__(self) -> None:
        for field in ("run_id", "cell_id", "stream_id"):
            _safe_id(getattr(self, field), field)
        try:
            canonical_uuid = str(uuid.UUID(self.session_uuid))
        except (ValueError, AttributeError, TypeError) as exc:
            raise IdentityError("session_uuid is not a UUID") from exc
        _require(canonical_uuid == self.session_uuid,
                 "session_uuid is not canonical", IdentityError)
        _digest(self.controller_lineage_sha256, "controller_lineage_sha256")
        _digest(self.execution_bundle_sha256, "execution_bundle_sha256")
        for field in ("decision_seq", "ticket_seq", "frame_id", "tensor_seq"):
            _nonnegative(getattr(self, field), field)
        _require(type(self.capture_timestamp_ns) is int
                 and self.capture_timestamp_ns > 0,
                 "capture_timestamp_ns must be a positive integer", IdentityError)
        _require(type(self.mode_id) is int and 0 <= self.mode_id < 12,
                 "mode_id is outside [0,11]", IdentityError)
        _require(type(self.q_e4) is int and 0 <= self.q_e4 <= 9800,
                 "q_e4 is outside [0,9800]", IdentityError)
        _require(type(self.keep_count) is int and self.keep_count > 0,
                 "keep_count must be a positive integer", IdentityError)
        paired = self.anchor_action_id is None, self.profile_id is None
        _require(paired[0] == paired[1],
                 "anchor_action_id and profile_id must both be null or present",
                 IdentityError)
        if self.anchor_action_id is not None:
            _require(type(self.anchor_action_id) is int
                     and 0 <= self.anchor_action_id < 72,
                     "anchor_action_id is outside [0,71]", IdentityError)
            _safe_id(self.profile_id, "profile_id")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrameActionIdentityV1":
        fields = set(cls.__dataclass_fields__)
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "identity fields are incomplete or foreign", IdentityError)
        return cls(**{field: raw[field] for field in fields})

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def canonical_bytes(self) -> bytes:
        return _canonical({"schema": IDENTITY_SCHEMA, **self.as_dict()})

    def exact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def decision_key(self) -> tuple[Any, ...]:
        """Lookup key broad enough to detect a conflicting exact identity."""
        return (
            self.run_id, self.cell_id, self.stream_id, self.session_uuid,
            self.controller_lineage_sha256, self.decision_seq, self.ticket_seq,
        )

    def postrun_join_key(self) -> dict[str, Any]:
        """Exact key for joining predictions to separately retained CARLA GT."""
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "stream_id": self.stream_id,
            "frame_id": self.frame_id,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "session_uuid": self.session_uuid,
            "controller_lineage_sha256": self.controller_lineage_sha256,
            "decision_seq": self.decision_seq,
            "ticket_seq": self.ticket_seq,
            "tensor_seq": self.tensor_seq,
            "action": {
                "mode_id": self.mode_id,
                "q_e4": self.q_e4,
                "keep_count": self.keep_count,
                "anchor_action_id": self.anchor_action_id,
                "profile_id": self.profile_id,
                "execution_bundle_sha256": self.execution_bundle_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class TailOutputAckV1:
    """Edge-to-UE acknowledgement emitted at usable tail-output readiness."""

    identity: FrameActionIdentityV1
    status: str
    tail_output_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.identity) is FrameActionIdentityV1,
                 "identity must be exactly FrameActionIdentityV1", IdentityError)
        _require(self.status == ACK_STATUS,
                 f"status must be {ACK_STATUS}", OperationalAckError)
        _digest(self.tail_output_sha256, "tail_output_sha256")

    @classmethod
    def success(cls, identity: FrameActionIdentityV1,
                tail_output: bytes) -> "TailOutputAckV1":
        _require(type(tail_output) is bytes and len(tail_output) > 0,
                 "tail_output must be non-empty bytes")
        return cls(identity=identity, status=ACK_STATUS,
                   tail_output_sha256=hashlib.sha256(tail_output).hexdigest())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TailOutputAckV1":
        _require(isinstance(raw, Mapping)
                 and set(raw) == {"schema", "identity", "status",
                                  "tail_output_sha256"},
                 "ACK fields are incomplete or foreign", AckWireError)
        _require(raw["schema"] == ACK_SCHEMA, "ACK schema drift", AckWireError)
        return cls(identity=FrameActionIdentityV1.from_mapping(raw["identity"]),
                   status=raw["status"],
                   tail_output_sha256=raw["tail_output_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ACK_SCHEMA,
            "identity": self.identity.as_dict(),
            "status": self.status,
            "tail_output_sha256": self.tail_output_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical(self.as_dict())


def encode_ack(ack: TailOutputAckV1) -> bytes:
    _require(type(ack) is TailOutputAckV1,
             "ack must be exactly TailOutputAckV1", AckWireError)
    payload = ack.canonical_bytes()
    header = _HEADER.pack(_MAGIC, len(payload))
    body = header + payload
    return body + hashlib.sha256(body).digest()


def decode_ack(packet: bytes) -> TailOutputAckV1:
    _require(type(packet) is bytes, "ACK packet must be bytes", AckWireError)
    _require(len(packet) >= _HEADER.size + 2 + _DIGEST_BYTES,
             "ACK packet is truncated", AckWireError)
    body, digest = packet[:-_DIGEST_BYTES], packet[-_DIGEST_BYTES:]
    _require(hashlib.sha256(body).digest() == digest,
             "ACK packet digest mismatch", AckWireError)
    magic, length = _HEADER.unpack(body[:_HEADER.size])
    _require(magic == _MAGIC, "ACK packet magic mismatch", AckWireError)
    payload = body[_HEADER.size:]
    _require(length == len(payload), "ACK packet length mismatch", AckWireError)
    try:
        raw = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AckWireError("ACK payload is not canonical JSON") from exc
    ack = TailOutputAckV1.from_mapping(raw)
    _require(ack.canonical_bytes() == payload,
             "ACK payload is not in canonical form", AckWireError)
    return ack


class AckClass(str, enum.Enum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE_IGNORED = "DUPLICATE_IGNORED"
    LATE_ORPHAN = "LATE_ORPHAN"
    UNKNOWN_ORPHAN = "UNKNOWN_ORPHAN"


class OperationalTerminal(str, enum.Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True)
class OperationalOutcomeV1:
    identity: FrameActionIdentityV1
    terminal: OperationalTerminal
    action_open_monotonic_raw_ns: int
    resolution_monotonic_raw_ns: int
    observed_latency_ns: Optional[int]
    state_latency_ns: int
    accepted_ack_sha256: Optional[str]
    tail_output_sha256: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.terminal is OperationalTerminal.SUCCESS

    @property
    def ack_receipt_monotonic_raw_ns(self) -> Optional[int]:
        """The accepted ACK receipt stamp; timeouts have no accepted ACK."""
        return (self.resolution_monotonic_raw_ns if self.success else None)


@dataclass(frozen=True, slots=True)
class DurableOperationalSnapshotV1:
    """Verified create-only evidence, ready for the post-run evaluator."""

    opened_identities: tuple[FrameActionIdentityV1, ...]
    outcomes: tuple[OperationalOutcomeV1, ...]
    late_orphans: tuple[Mapping[str, Any], ...]
    unknown_orphans: tuple[Mapping[str, Any], ...]


class OperationalEvidenceStoreV1:
    """Crash-local, create-only UE evidence for operational policy outcomes.

    Prediction payload bytes are intentionally not duplicated here.  The edge
    retains those in ``PredictionEvidenceStoreV1`` under the same exact
    identity and tail-output digest.  This store owns the UE-clock facts:
    ticket open, accepted ACK receipt/outcome, and late/unknown ACK arrivals.
    """

    _DIRECTORIES = ("opened", "outcomes", "late_orphans", "unknown_orphans")

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.opened = self.root / "opened"
        self.outcomes = self.root / "outcomes"
        self.late = self.root / "late_orphans"
        self.unknown = self.root / "unknown_orphans"
        self._lock = threading.Lock()

    @classmethod
    def create(cls, root: Path) -> "OperationalEvidenceStoreV1":
        store = cls(root)
        try:
            store.root.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise OperationalCreateOnlyError(
                f"operational evidence root already exists: {store.root}") from exc
        for name in cls._DIRECTORIES:
            (store.root / name).mkdir()
        return store

    @classmethod
    def open_existing(cls, root: Path) -> "OperationalEvidenceStoreV1":
        store = cls(root)
        _require(store.root.is_dir() and not store.root.is_symlink(),
                 "operational evidence root is missing or a symlink",
                 OperationalEvidenceError)
        for name in cls._DIRECTORIES:
            path = store.root / name
            _require(path.is_dir() and not path.is_symlink(),
                     f"operational evidence directory is missing: {name}",
                     OperationalEvidenceError)
        return store

    @staticmethod
    def _write_exclusive(path: Path, raw: Mapping[str, Any]) -> None:
        payload = _canonical(raw) + b"\n"
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise OperationalCreateOnlyError(
                f"create-only operational evidence exists: {path}") from exc

    @staticmethod
    def _read_canonical(path: Path) -> dict[str, Any]:
        _require(path.is_file() and not path.is_symlink()
                 and path.suffix == ".json",
                 "operational evidence entry is foreign or a symlink",
                 OperationalEvidenceError)
        payload = path.read_bytes()
        try:
            raw = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperationalEvidenceError(
                "operational evidence is not valid JSON") from exc
        _require(type(raw) is dict and payload == _canonical(raw) + b"\n",
                 "operational evidence is not canonical",
                 OperationalEvidenceError)
        return raw

    @staticmethod
    def _identity(raw: Mapping[str, Any]) -> FrameActionIdentityV1:
        identity = FrameActionIdentityV1.from_mapping(raw["identity"])
        _require(raw["identity_sha256"] == identity.exact_sha256(),
                 "operational identity digest mismatch",
                 OperationalEvidenceError)
        return identity

    def record_open(self, identity: FrameActionIdentityV1,
                    action_open_monotonic_raw_ns: int) -> None:
        _nonnegative(action_open_monotonic_raw_ns,
                     "action_open_monotonic_raw_ns")
        identity_sha = identity.exact_sha256()
        raw = {
            "schema": OPEN_RECORD_SCHEMA,
            "identity": identity.as_dict(),
            "identity_sha256": identity_sha,
            "clock_domain": CLOCK_DOMAIN,
            "action_open_monotonic_raw_ns": action_open_monotonic_raw_ns,
            "inclusive_deadline_ns": ACK_DEADLINE_NS,
            "deadline_monotonic_raw_ns": (
                action_open_monotonic_raw_ns + ACK_DEADLINE_NS),
        }
        with self._lock:
            self._write_exclusive(self.opened / f"{identity_sha}.json", raw)

    def record_outcome(self, outcome: OperationalOutcomeV1) -> None:
        _require(type(outcome) is OperationalOutcomeV1,
                 "outcome must be exactly OperationalOutcomeV1",
                 OperationalEvidenceError)
        identity_sha = outcome.identity.exact_sha256()
        raw = {
            "schema": OUTCOME_RECORD_SCHEMA,
            "identity": outcome.identity.as_dict(),
            "identity_sha256": identity_sha,
            "clock_domain": CLOCK_DOMAIN,
            "terminal": outcome.terminal.value,
            "success": outcome.success,
            "action_open_monotonic_raw_ns": (
                outcome.action_open_monotonic_raw_ns),
            "resolution_monotonic_raw_ns": (
                outcome.resolution_monotonic_raw_ns),
            "ack_receipt_monotonic_raw_ns": (
                outcome.ack_receipt_monotonic_raw_ns),
            "observed_operational_latency_ns": outcome.observed_latency_ns,
            "state_latency_ns": outcome.state_latency_ns,
            "accepted_ack_sha256": outcome.accepted_ack_sha256,
            "tail_output_sha256": outcome.tail_output_sha256,
            "inclusive_deadline_ns": ACK_DEADLINE_NS,
        }
        with self._lock:
            self._write_exclusive(self.outcomes / f"{identity_sha}.json", raw)

    @staticmethod
    def _orphan_raw(*, kind: str, identity: FrameActionIdentityV1,
                    receipt_monotonic_raw_ns: int,
                    observed_latency_ns: Optional[int],
                    ack_sha256: str, tail_output_sha256: str) -> dict[str, Any]:
        _require(kind in {"LATE", "UNKNOWN"}, "unknown orphan kind",
                 OperationalEvidenceError)
        _nonnegative(receipt_monotonic_raw_ns,
                     "receipt_monotonic_raw_ns")
        if observed_latency_ns is not None:
            _nonnegative(observed_latency_ns, "observed_latency_ns")
        _digest(ack_sha256, "ack_sha256")
        _digest(tail_output_sha256, "tail_output_sha256")
        return {
            "schema": ORPHAN_RECORD_SCHEMA,
            "kind": kind,
            "identity": identity.as_dict(),
            "identity_sha256": identity.exact_sha256(),
            "clock_domain": CLOCK_DOMAIN,
            "receipt_monotonic_raw_ns": receipt_monotonic_raw_ns,
            "observed_latency_ns": observed_latency_ns,
            "ack_sha256": ack_sha256,
            "tail_output_sha256": tail_output_sha256,
        }

    def record_orphan(self, *, kind: str, identity: FrameActionIdentityV1,
                      receipt_monotonic_raw_ns: int,
                      observed_latency_ns: Optional[int],
                      ack_sha256: str, tail_output_sha256: str) -> None:
        raw = self._orphan_raw(
            kind=kind, identity=identity,
            receipt_monotonic_raw_ns=receipt_monotonic_raw_ns,
            observed_latency_ns=observed_latency_ns,
            ack_sha256=ack_sha256,
            tail_output_sha256=tail_output_sha256)
        directory = self.late if kind == "LATE" else self.unknown
        name = (f"{identity.exact_sha256()}__{receipt_monotonic_raw_ns:020d}"
                f"__{ack_sha256}.json")
        with self._lock:
            self._write_exclusive(directory / name, raw)

    @staticmethod
    def _parse_open(raw: Mapping[str, Any]) -> tuple[FrameActionIdentityV1, int]:
        fields = {
            "schema", "identity", "identity_sha256", "clock_domain",
            "action_open_monotonic_raw_ns", "inclusive_deadline_ns",
            "deadline_monotonic_raw_ns",
        }
        _require(set(raw) == fields and raw["schema"] == OPEN_RECORD_SCHEMA,
                 "open record schema/fields drifted", OperationalEvidenceError)
        identity = OperationalEvidenceStoreV1._identity(raw)
        opened = _nonnegative(raw["action_open_monotonic_raw_ns"],
                              "action_open_monotonic_raw_ns")
        _require(raw["clock_domain"] == CLOCK_DOMAIN
                 and raw["inclusive_deadline_ns"] == ACK_DEADLINE_NS
                 and raw["deadline_monotonic_raw_ns"]
                 == opened + ACK_DEADLINE_NS,
                 "open record clock/deadline drifted",
                 OperationalEvidenceError)
        return identity, opened

    @staticmethod
    def _parse_outcome(raw: Mapping[str, Any]) -> OperationalOutcomeV1:
        fields = {
            "schema", "identity", "identity_sha256", "clock_domain",
            "terminal", "success", "action_open_monotonic_raw_ns",
            "resolution_monotonic_raw_ns", "ack_receipt_monotonic_raw_ns",
            "observed_operational_latency_ns", "state_latency_ns",
            "accepted_ack_sha256", "tail_output_sha256",
            "inclusive_deadline_ns",
        }
        _require(set(raw) == fields and raw["schema"] == OUTCOME_RECORD_SCHEMA,
                 "outcome record schema/fields drifted",
                 OperationalEvidenceError)
        identity = OperationalEvidenceStoreV1._identity(raw)
        try:
            terminal = OperationalTerminal(raw["terminal"])
        except (TypeError, ValueError) as exc:
            raise OperationalEvidenceError("unknown operational terminal") from exc
        outcome = OperationalOutcomeV1(
            identity=identity,
            terminal=terminal,
            action_open_monotonic_raw_ns=raw["action_open_monotonic_raw_ns"],
            resolution_monotonic_raw_ns=raw["resolution_monotonic_raw_ns"],
            observed_latency_ns=raw["observed_operational_latency_ns"],
            state_latency_ns=raw["state_latency_ns"],
            accepted_ack_sha256=raw["accepted_ack_sha256"],
            tail_output_sha256=raw["tail_output_sha256"],
        )
        _require(raw["clock_domain"] == CLOCK_DOMAIN
                 and raw["inclusive_deadline_ns"] == ACK_DEADLINE_NS
                 and raw["success"] is outcome.success,
                 "outcome clock/deadline/status drifted",
                 OperationalEvidenceError)
        opened = _nonnegative(outcome.action_open_monotonic_raw_ns,
                              "action_open_monotonic_raw_ns")
        resolution = _nonnegative(outcome.resolution_monotonic_raw_ns,
                                  "resolution_monotonic_raw_ns")
        if outcome.success:
            latency = outcome.observed_latency_ns
            _require(type(latency) is int and 0 <= latency <= ACK_DEADLINE_NS,
                     "successful operational latency is invalid",
                     OperationalEvidenceError)
            _require(resolution == opened + latency
                     and raw["ack_receipt_monotonic_raw_ns"] == resolution
                     and outcome.state_latency_ns == latency,
                     "successful outcome timestamps/state latency drifted",
                     OperationalEvidenceError)
            _digest(outcome.accepted_ack_sha256, "accepted_ack_sha256")
            _digest(outcome.tail_output_sha256, "tail_output_sha256")
        else:
            _require(outcome.observed_latency_ns is None
                     and raw["ack_receipt_monotonic_raw_ns"] is None
                     and resolution == opened + FIRST_LATE_TICK_NS
                     and outcome.state_latency_ns == 0
                     and outcome.accepted_ack_sha256 is None
                     and outcome.tail_output_sha256 is None,
                     "timeout outcome is not correctly censored",
                     OperationalEvidenceError)
        return outcome

    @staticmethod
    def _parse_orphan(raw: Mapping[str, Any]) -> dict[str, Any]:
        fields = {
            "schema", "kind", "identity", "identity_sha256",
            "clock_domain", "receipt_monotonic_raw_ns",
            "observed_latency_ns", "ack_sha256", "tail_output_sha256",
        }
        _require(set(raw) == fields and raw["schema"] == ORPHAN_RECORD_SCHEMA
                 and raw["kind"] in {"LATE", "UNKNOWN"}
                 and raw["clock_domain"] == CLOCK_DOMAIN,
                 "orphan record schema/fields drifted",
                 OperationalEvidenceError)
        identity = OperationalEvidenceStoreV1._identity(raw)
        receipt = _nonnegative(raw["receipt_monotonic_raw_ns"],
                               "receipt_monotonic_raw_ns")
        observed = raw["observed_latency_ns"]
        if observed is not None:
            _nonnegative(observed, "observed_latency_ns")
        _digest(raw["ack_sha256"], "ack_sha256")
        _digest(raw["tail_output_sha256"], "tail_output_sha256")
        result = dict(raw)
        result["identity"] = identity
        result["receipt_monotonic_raw_ns"] = receipt
        return result

    def verify_all(self, *, require_all_resolved: bool = False
                   ) -> DurableOperationalSnapshotV1:
        _require(type(require_all_resolved) is bool,
                 "require_all_resolved must be bool", OperationalEvidenceError)
        self.open_existing(self.root)
        _require({path.name for path in self.root.iterdir()}
                 == set(self._DIRECTORIES),
                 "foreign path in operational evidence root",
                 OperationalEvidenceError)
        opens: dict[str, tuple[FrameActionIdentityV1, int]] = {}
        decision_keys: dict[tuple[Any, ...], str] = {}
        for path in sorted(self.opened.iterdir()):
            raw = self._read_canonical(path)
            identity, opened = self._parse_open(raw)
            identity_sha = identity.exact_sha256()
            _require(path.name == f"{identity_sha}.json",
                     "open filename differs from identity",
                     OperationalEvidenceError)
            prior = decision_keys.get(identity.decision_key())
            _require(prior in (None, identity_sha),
                     "one logical decision has conflicting durable identities",
                     OperationalEvidenceError)
            decision_keys[identity.decision_key()] = identity_sha
            opens[identity_sha] = (identity, opened)

        outcomes: dict[str, OperationalOutcomeV1] = {}
        for path in sorted(self.outcomes.iterdir()):
            raw = self._read_canonical(path)
            outcome = self._parse_outcome(raw)
            identity_sha = outcome.identity.exact_sha256()
            _require(path.name == f"{identity_sha}.json"
                     and identity_sha in opens
                     and opens[identity_sha][0] == outcome.identity
                     and opens[identity_sha][1]
                     == outcome.action_open_monotonic_raw_ns,
                     "outcome does not reconcile with its durable open",
                     OperationalEvidenceError)
            outcomes[identity_sha] = outcome
        if require_all_resolved:
            _require(set(outcomes) == set(opens),
                     "not every opened policy decision has a durable outcome",
                     OperationalEvidenceError)

        late: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        for directory, kind, destination in (
            (self.late, "LATE", late),
            (self.unknown, "UNKNOWN", unknown),
        ):
            for path in sorted(directory.iterdir()):
                raw = self._read_canonical(path)
                parsed = self._parse_orphan(raw)
                identity = parsed["identity"]
                identity_sha = identity.exact_sha256()
                expected_name = (
                    f"{identity_sha}__{parsed['receipt_monotonic_raw_ns']:020d}"
                    f"__{parsed['ack_sha256']}.json")
                _require(path.name == expected_name and parsed["kind"] == kind,
                         "orphan filename/kind drifted",
                         OperationalEvidenceError)
                if kind == "LATE":
                    _require(identity_sha in outcomes
                             and outcomes[identity_sha].terminal
                             is OperationalTerminal.TIMEOUT
                             and parsed["observed_latency_ns"]
                             == (parsed["receipt_monotonic_raw_ns"]
                                 - opens[identity_sha][1])
                             and parsed["observed_latency_ns"]
                             > ACK_DEADLINE_NS,
                             "late orphan does not reconcile with timeout",
                             OperationalEvidenceError)
                destination.append(parsed)
        return DurableOperationalSnapshotV1(
            opened_identities=tuple(opens[key][0] for key in sorted(opens)),
            outcomes=tuple(outcomes[key] for key in sorted(outcomes)),
            late_orphans=tuple(late),
            unknown_orphans=tuple(unknown),
        )

    def load_outcomes(self, *, require_all_resolved: bool = True
                      ) -> tuple[OperationalOutcomeV1, ...]:
        return self.verify_all(
            require_all_resolved=require_all_resolved).outcomes


@dataclass(slots=True)
class _Ticket:
    identity: FrameActionIdentityV1
    action_open_ns: int
    outcome: Optional[OperationalOutcomeV1] = None
    accepted_ack_bytes: Optional[bytes] = None


class OperationalAckLedgerV1:
    """UE-local exact-ticket ledger; all latency arithmetic is one-clock."""

    def __init__(self, *,
                 evidence_store: Optional[OperationalEvidenceStoreV1] = None
                 ) -> None:
        _require(evidence_store is None
                 or type(evidence_store) is OperationalEvidenceStoreV1,
                 "evidence_store must be OperationalEvidenceStoreV1 or None",
                 OperationalEvidenceError)
        self._tickets: dict[tuple[Any, ...], _Ticket] = {}
        self.late_orphans: list[dict[str, Any]] = []
        self.unknown_orphans: list[dict[str, Any]] = []
        self._evidence_store = evidence_store
        self._lock = threading.RLock()

    def open(self, identity: FrameActionIdentityV1,
             action_open_monotonic_raw_ns: int) -> None:
        _require(type(identity) is FrameActionIdentityV1,
                 "identity must be exactly FrameActionIdentityV1", IdentityError)
        _nonnegative(action_open_monotonic_raw_ns,
                     "action_open_monotonic_raw_ns")
        with self._lock:
            key = identity.decision_key()
            if key in self._tickets:
                existing = self._tickets[key]
                if existing.identity != identity:
                    raise AckIdentityConflict(
                        "decision key already has another identity")
                raise OperationalAckError("ticket already exists")
            if self._evidence_store is not None:
                self._evidence_store.record_open(
                    identity, action_open_monotonic_raw_ns)
            self._tickets[key] = _Ticket(identity, action_open_monotonic_raw_ns)

    def _timeout(self, ticket: _Ticket) -> None:
        if ticket.outcome is not None:
            return
        outcome = OperationalOutcomeV1(
            identity=ticket.identity,
            terminal=OperationalTerminal.TIMEOUT,
            action_open_monotonic_raw_ns=ticket.action_open_ns,
            resolution_monotonic_raw_ns=(ticket.action_open_ns
                                         + FIRST_LATE_TICK_NS),
            observed_latency_ns=None,
            # Run-4B/5B masks this slot with
            # ``prev_operational_success == 0``.  Keep the state value at the
            # registered failure sentinel instead of fabricating a successful
            # 170-ms latency observation.
            state_latency_ns=0,
            accepted_ack_sha256=None,
            tail_output_sha256=None,
        )
        if self._evidence_store is not None:
            self._evidence_store.record_outcome(outcome)
        ticket.outcome = outcome

    def poll(self, now_monotonic_raw_ns: int) -> tuple[FrameActionIdentityV1, ...]:
        _nonnegative(now_monotonic_raw_ns, "now_monotonic_raw_ns")
        with self._lock:
            newly_timed_out: list[FrameActionIdentityV1] = []
            for ticket in self._tickets.values():
                _require(now_monotonic_raw_ns >= ticket.action_open_ns,
                         "clock moved before action-open")
                if (ticket.outcome is None
                        and now_monotonic_raw_ns
                        > ticket.action_open_ns + ACK_DEADLINE_NS):
                    self._timeout(ticket)
                    newly_timed_out.append(ticket.identity)
            return tuple(newly_timed_out)

    def receive(self, ack_or_packet: TailOutputAckV1 | bytes,
                receipt_monotonic_raw_ns: int) -> AckClass:
        _nonnegative(receipt_monotonic_raw_ns, "receipt_monotonic_raw_ns")
        ack = (decode_ack(ack_or_packet) if type(ack_or_packet) is bytes
               else ack_or_packet)
        _require(type(ack) is TailOutputAckV1,
                 "ack must be TailOutputAckV1 or bytes", AckWireError)
        ack_bytes = ack.canonical_bytes()
        ack_sha = hashlib.sha256(ack_bytes).hexdigest()
        with self._lock:
            key = ack.identity.decision_key()
            ticket = self._tickets.get(key)
            if ticket is None:
                if self._evidence_store is not None:
                    self._evidence_store.record_orphan(
                        kind="UNKNOWN", identity=ack.identity,
                        receipt_monotonic_raw_ns=receipt_monotonic_raw_ns,
                        observed_latency_ns=None,
                        ack_sha256=ack_sha,
                        tail_output_sha256=ack.tail_output_sha256)
                self.unknown_orphans.append({
                    "identity_sha256": ack.identity.exact_sha256(),
                    "receipt_monotonic_raw_ns": receipt_monotonic_raw_ns,
                    "ack_sha256": ack_sha,
                    "tail_output_sha256": ack.tail_output_sha256,
                })
                return AckClass.UNKNOWN_ORPHAN
            if ticket.identity != ack.identity:
                raise AckIdentityConflict(
                    "ACK logical decision matches but exact frame/action identity differs")
            _require(receipt_monotonic_raw_ns >= ticket.action_open_ns,
                     "ACK receipt precedes action-open")
            if ticket.accepted_ack_bytes is not None:
                if ack_bytes == ticket.accepted_ack_bytes:
                    return AckClass.DUPLICATE_IGNORED
                raise ConflictingAckError(
                    "non-identical ACK duplicates one accepted ticket")
            latency = receipt_monotonic_raw_ns - ticket.action_open_ns
            if ticket.outcome is not None:
                _require(ticket.outcome.terminal is OperationalTerminal.TIMEOUT,
                         "resolved ticket lacks accepted ACK bytes")
                if self._evidence_store is not None:
                    self._evidence_store.record_orphan(
                        kind="LATE", identity=ack.identity,
                        receipt_monotonic_raw_ns=receipt_monotonic_raw_ns,
                        observed_latency_ns=latency,
                        ack_sha256=ack_sha,
                        tail_output_sha256=ack.tail_output_sha256)
                self.late_orphans.append({
                    "identity_sha256": ack.identity.exact_sha256(),
                    "receipt_monotonic_raw_ns": receipt_monotonic_raw_ns,
                    "observed_late_latency_ns": latency,
                    "ack_sha256": ack_sha,
                    "tail_output_sha256": ack.tail_output_sha256,
                })
                return AckClass.LATE_ORPHAN
            if latency <= ACK_DEADLINE_NS:
                outcome = OperationalOutcomeV1(
                    identity=ticket.identity,
                    terminal=OperationalTerminal.SUCCESS,
                    action_open_monotonic_raw_ns=ticket.action_open_ns,
                    resolution_monotonic_raw_ns=receipt_monotonic_raw_ns,
                    observed_latency_ns=latency,
                    state_latency_ns=latency,
                    accepted_ack_sha256=ack_sha,
                    tail_output_sha256=ack.tail_output_sha256,
                )
                if self._evidence_store is not None:
                    self._evidence_store.record_outcome(outcome)
                ticket.accepted_ack_bytes = ack_bytes
                ticket.outcome = outcome
                return AckClass.ACCEPTED
            self._timeout(ticket)
            if self._evidence_store is not None:
                self._evidence_store.record_orphan(
                    kind="LATE", identity=ack.identity,
                    receipt_monotonic_raw_ns=receipt_monotonic_raw_ns,
                    observed_latency_ns=latency,
                    ack_sha256=ack_sha,
                    tail_output_sha256=ack.tail_output_sha256)
            self.late_orphans.append({
                "identity_sha256": ack.identity.exact_sha256(),
                "receipt_monotonic_raw_ns": receipt_monotonic_raw_ns,
                "observed_late_latency_ns": latency,
                "ack_sha256": ack_sha,
                "tail_output_sha256": ack.tail_output_sha256,
            })
            return AckClass.LATE_ORPHAN

    def outcome(self, identity: FrameActionIdentityV1) -> Optional[OperationalOutcomeV1]:
        with self._lock:
            ticket = self._tickets.get(identity.decision_key())
            if ticket is None:
                raise OperationalAckError("unknown ticket")
            if ticket.identity != identity:
                raise AckIdentityConflict("ticket identity differs")
            return ticket.outcome

    def resolved_outcomes(self) -> tuple[OperationalOutcomeV1, ...]:
        """Return the immutable resolved UE outcomes in insertion order."""
        with self._lock:
            return tuple(ticket.outcome for ticket in self._tickets.values()
                         if ticket.outcome is not None)

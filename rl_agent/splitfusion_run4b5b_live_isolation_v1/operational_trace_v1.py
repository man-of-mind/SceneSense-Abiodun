"""Durable create-only operational outcomes for post-run evaluation.

This is the primary (superset) frame population.  In particular, a timed-out
decision remains a valid operational record even when the edge never produced
a tail output.  Prediction and CARLA evidence are optional downstream joins;
neither is permitted to erase an operational row.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .operational_ack_v1 import (
    ACK_DEADLINE_NS,
    FIRST_LATE_TICK_NS,
    FrameActionIdentityV1,
    OperationalOutcomeV1,
    OperationalTerminal,
)


TRACE_RECORD_SCHEMA = "scenesense.splitfusion.run4b5b.operational_trace_record.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OperationalTraceError(RuntimeError):
    """A durable operational trace is invalid or inconsistent."""


class OperationalTraceCreateOnlyError(OperationalTraceError):
    """A create-only trace path already exists."""


class OperationalTraceIntegrityError(OperationalTraceError):
    """A trace record is missing, foreign, or non-canonical."""


def _require(condition: bool, message: str,
             error: type[OperationalTraceError] = OperationalTraceError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise OperationalTraceIntegrityError(
            "operational trace is not canonicalizable"
        ) from exc


def _sha256(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SHA256_RE.fullmatch(value)),
             f"{field} is not a lowercase SHA-256",
             OperationalTraceIntegrityError)
    return value


@dataclass(frozen=True, slots=True)
class OperationalTraceRecordV1:
    outcome: OperationalOutcomeV1
    identity_sha256: str
    payload_bytes: int

    def __post_init__(self) -> None:
        _require(type(self.outcome) is OperationalOutcomeV1,
                 "outcome must be exactly OperationalOutcomeV1",
                 OperationalTraceIntegrityError)
        _sha256(self.identity_sha256, "identity_sha256")
        _require(self.identity_sha256 == self.outcome.identity.exact_sha256(),
                 "operational identity digest mismatch",
                 OperationalTraceIntegrityError)
        _require(type(self.payload_bytes) is int and self.payload_bytes > 0,
                 "payload_bytes must be a positive exact integer",
                 OperationalTraceIntegrityError)
        value = self.outcome
        for field in (
            "action_open_monotonic_raw_ns", "resolution_monotonic_raw_ns",
            "state_latency_ns",
        ):
            _require(type(getattr(value, field)) is int
                     and getattr(value, field) >= 0,
                     f"{field} must be nonnegative",
                     OperationalTraceIntegrityError)
        _require(value.resolution_monotonic_raw_ns
                 >= value.action_open_monotonic_raw_ns,
                 "operational resolution precedes action-open",
                 OperationalTraceIntegrityError)
        if value.terminal is OperationalTerminal.SUCCESS:
            _require(type(value.observed_latency_ns) is int
                     and 0 <= value.observed_latency_ns <= ACK_DEADLINE_NS,
                     "successful trace lacks an on-time observed latency",
                     OperationalTraceIntegrityError)
            _require(value.state_latency_ns == value.observed_latency_ns,
                     "successful state latency differs from observed latency",
                     OperationalTraceIntegrityError)
            _sha256(value.accepted_ack_sha256, "accepted_ack_sha256")
            _require(value.resolution_monotonic_raw_ns
                     == value.action_open_monotonic_raw_ns
                     + value.observed_latency_ns,
                     "successful trace timing does not reconcile",
                     OperationalTraceIntegrityError)
        elif value.terminal is OperationalTerminal.TIMEOUT:
            _require(value.observed_latency_ns is None,
                     "timeout carries a fabricated observed latency",
                     OperationalTraceIntegrityError)
            _require(value.state_latency_ns == 0,
                     "timeout state latency must use the masked zero sentinel",
                     OperationalTraceIntegrityError)
            _require(value.accepted_ack_sha256 is None,
                     "timeout carries an accepted ACK",
                     OperationalTraceIntegrityError)
            _require(value.resolution_monotonic_raw_ns
                     == value.action_open_monotonic_raw_ns + FIRST_LATE_TICK_NS,
                     "timeout resolution differs from the inclusive boundary",
                     OperationalTraceIntegrityError)
        else:  # pragma: no cover - enum is closed, retained defensively
            raise OperationalTraceIntegrityError("unknown operational terminal")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OperationalTraceRecordV1":
        fields = {
            "schema", "identity", "identity_sha256", "payload_bytes",
            "terminal", "action_open_monotonic_raw_ns",
            "resolution_monotonic_raw_ns", "observed_latency_ns",
            "state_latency_ns", "accepted_ack_sha256",
        }
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "operational trace fields are incomplete or foreign",
                 OperationalTraceIntegrityError)
        _require(raw["schema"] == TRACE_RECORD_SCHEMA,
                 "operational trace schema drift",
                 OperationalTraceIntegrityError)
        try:
            terminal = OperationalTerminal(raw["terminal"])
        except (TypeError, ValueError) as exc:
            raise OperationalTraceIntegrityError(
                "unknown operational terminal"
            ) from exc
        outcome = OperationalOutcomeV1(
            identity=FrameActionIdentityV1.from_mapping(raw["identity"]),
            terminal=terminal,
            action_open_monotonic_raw_ns=raw["action_open_monotonic_raw_ns"],
            resolution_monotonic_raw_ns=raw["resolution_monotonic_raw_ns"],
            observed_latency_ns=raw["observed_latency_ns"],
            state_latency_ns=raw["state_latency_ns"],
            accepted_ack_sha256=raw["accepted_ack_sha256"],
        )
        return cls(
            outcome=outcome,
            identity_sha256=raw["identity_sha256"],
            payload_bytes=raw["payload_bytes"],
        )

    def as_dict(self) -> dict[str, Any]:
        value = self.outcome
        return {
            "schema": TRACE_RECORD_SCHEMA,
            "identity": value.identity.as_dict(),
            "identity_sha256": self.identity_sha256,
            "payload_bytes": self.payload_bytes,
            "terminal": value.terminal.value,
            "action_open_monotonic_raw_ns": value.action_open_monotonic_raw_ns,
            "resolution_monotonic_raw_ns": value.resolution_monotonic_raw_ns,
            "observed_latency_ns": value.observed_latency_ns,
            "state_latency_ns": value.state_latency_ns,
            "accepted_ack_sha256": value.accepted_ack_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical(self.as_dict()) + b"\n"


class OperationalTraceStoreV1:
    """Create-only one-record-per-policy-decision operational ledger."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.records = self.root / "records"

    @classmethod
    def create(cls, root: Path) -> "OperationalTraceStoreV1":
        store = cls(root)
        try:
            store.root.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise OperationalTraceCreateOnlyError(
                f"operational trace root already exists: {store.root}"
            ) from exc
        store.records.mkdir()
        return store

    @classmethod
    def open_existing(cls, root: Path) -> "OperationalTraceStoreV1":
        store = cls(root)
        _require(store.root.is_dir() and store.records.is_dir(),
                 "operational trace store is incomplete",
                 OperationalTraceIntegrityError)
        return store

    @staticmethod
    def _write_exclusive(path: Path, payload: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise OperationalTraceCreateOnlyError(
                f"operational record already exists: {path}"
            ) from exc

    def write(self, outcome: OperationalOutcomeV1,
              payload_bytes: int) -> OperationalTraceRecordV1:
        _require(type(outcome) is OperationalOutcomeV1,
                 "outcome must be exactly OperationalOutcomeV1")
        digest = outcome.identity.exact_sha256()
        record = OperationalTraceRecordV1(outcome, digest, payload_bytes)
        self._write_exclusive(
            self.records / f"{digest}.json", record.canonical_bytes())
        return record

    def verify_all(self) -> tuple[OperationalTraceRecordV1, ...]:
        _require(self.root.is_dir() and self.records.is_dir(),
                 "operational trace store is incomplete",
                 OperationalTraceIntegrityError)
        _require(not self.root.is_symlink() and not self.records.is_symlink(),
                 "operational trace directories may not be symlinks",
                 OperationalTraceIntegrityError)
        _require(not [path for path in self.root.iterdir()
                      if path.name != "records"],
                 "foreign path in operational trace root",
                 OperationalTraceIntegrityError)
        values: list[OperationalTraceRecordV1] = []
        logical: dict[tuple[Any, ...], str] = {}
        for path in sorted(self.records.iterdir()):
            _require(path.is_file() and not path.is_symlink()
                     and path.suffix == ".json",
                     "foreign operational record entry",
                     OperationalTraceIntegrityError)
            try:
                raw_bytes = path.read_bytes()
                raw = json.loads(raw_bytes.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OperationalTraceIntegrityError(
                    "operational record is invalid JSON"
                ) from exc
            record = OperationalTraceRecordV1.from_mapping(raw)
            _require(path.name == f"{record.identity_sha256}.json",
                     "operational filename differs from identity",
                     OperationalTraceIntegrityError)
            _require(raw_bytes == record.canonical_bytes(),
                     "operational record is not canonical",
                     OperationalTraceIntegrityError)
            key = record.outcome.identity.decision_key()
            prior = logical.get(key)
            _require(prior is None or prior == record.identity_sha256,
                     "one logical decision has conflicting exact identities",
                     OperationalTraceIntegrityError)
            logical[key] = record.identity_sha256
            values.append(record)
        return tuple(values)


__all__ = [
    "TRACE_RECORD_SCHEMA", "OperationalTraceError",
    "OperationalTraceCreateOnlyError", "OperationalTraceIntegrityError",
    "OperationalTraceRecordV1", "OperationalTraceStoreV1",
]

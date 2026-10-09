"""Independent UE ledger for privileged non-terminal quality outcomes."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Mapping

from . import protocol


FIELDS = (
    *protocol.IDENTITY_FIELDS,
    "capture_id", "registered_wall_ns", "ack_emit_start_wall_ns",
    "ack_ue_receive_wall_ns", "ack_edge_to_ue_ms", "capture_to_quality_ack_ms",
    "evaluator_mode", "event", "failure_reason", "detail_sha256",
    "message_sha256", "message_bytes", "source_address", "quality_json",
    "timing_json", "quality_ticket_terminal", "map_terminal",
    "duplicate_identical", "eligibility",
)


class QualityLedgerError(RuntimeError):
    pass


class QualityFeedbackLedger:
    """Exactly one quality disposition per sent frame, separate from map state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(FIELDS))
        self._writer.writeheader()
        self._handle.flush()
        self._lock = threading.Lock()
        self._pending: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._closed: dict[tuple[Any, ...], str] = {}
        self.identical_duplicates = 0
        self.not_eligible = 0

    @staticmethod
    def _key_from_identity(identity: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(identity["run_id"]), str(identity["cell_id"]),
            str(identity["stream_id"]), int(identity["frame_id"]),
            int(identity["action_id"]), str(identity["profile_id"]),
            int(identity["capture_timestamp_ns"]),
        )

    def register(self, identity: Mapping[str, Any], *, registered_wall_ns: int) -> None:
        record = {name: identity[name] for name in protocol.IDENTITY_FIELDS}
        key = self._key_from_identity(record)
        record["registered_wall_ns"] = int(registered_wall_ns)
        with self._lock:
            if key in self._pending or key in self._closed:
                raise QualityLedgerError(f"duplicate quality obligation: {key}")
            self._pending[key] = record

    def record(
        self,
        message: Mapping[str, Any],
        *, received_wall_ns: int,
        message_bytes: int,
        source_address: str,
    ) -> dict[str, Any]:
        protocol.validate(message)
        identity = protocol.identity_dict(message)
        key = self._key_from_identity(identity)
        message_digest = protocol.digest(message)
        with self._lock:
            if key in self._closed:
                if self._closed[key] != message_digest:
                    raise QualityLedgerError(f"conflicting quality duplicate: {key}")
                self.identical_duplicates += 1
                row = self._row(
                    identity, message, received_wall_ns=received_wall_ns,
                    message_bytes=message_bytes, source_address=source_address,
                    digest=message_digest, duplicate=True,
                    registered_wall_ns="",
                )
                self._append(row)
                return row
            base = self._pending.pop(key, None)
            if base is None:
                raise QualityLedgerError(f"quality ACK for unknown obligation: {key}")
            row = self._row(
                identity, message, received_wall_ns=received_wall_ns,
                message_bytes=message_bytes, source_address=source_address,
                digest=message_digest, duplicate=False,
                registered_wall_ns=base["registered_wall_ns"],
            )
            self._closed[key] = message_digest
            self._append(row)
            return row

    @staticmethod
    def _row(
        identity: Mapping[str, Any], message: Mapping[str, Any], *,
        received_wall_ns: int, message_bytes: int, source_address: str,
        digest: str, duplicate: bool, registered_wall_ns: Any,
    ) -> dict[str, Any]:
        timing = protocol.timing_dict(message)
        ack_emit = int(timing["ack_emit_start_wall_ns"] or 0)
        capture = int(identity["capture_timestamp_ns"])
        if received_wall_ns < ack_emit or received_wall_ns < capture:
            raise QualityLedgerError("negative quality-feedback interval")
        return {
            **identity,
            "capture_id": f"{identity['stream_id']}:{int(identity['frame_id'])}",
            "registered_wall_ns": registered_wall_ns,
            "ack_emit_start_wall_ns": ack_emit,
            "ack_ue_receive_wall_ns": int(received_wall_ns),
            "ack_edge_to_ue_ms": (int(received_wall_ns) - ack_emit) / 1e6,
            "capture_to_quality_ack_ms": (int(received_wall_ns) - capture) / 1e6,
            "evaluator_mode": str(message["m"]),
            "event": "QUALITY_EVALUATED" if message["e"] == "OK" else "QUALITY_EVALUATION_FAILED",
            "failure_reason": str(message.get("r") or ""),
            "detail_sha256": str(message["dh"]),
            "message_sha256": digest,
            "message_bytes": int(message_bytes),
            "source_address": str(source_address),
            "quality_json": json.dumps(
                protocol.quality_dict(message), sort_keys=True, separators=(",", ":")
            ),
            "timing_json": json.dumps(timing, sort_keys=True, separators=(",", ":")),
            "quality_ticket_terminal": True,
            "map_terminal": False,
            "duplicate_identical": bool(duplicate),
            "eligibility": "FINAL_PREDICTION_ELIGIBLE",
        }

    def mark_not_eligible(self, identity: Mapping[str, Any], *, reason: str) -> None:
        key = self._key_from_identity(identity)
        with self._lock:
            base = self._pending.pop(key, None)
            if base is None:
                if key in self._closed:
                    raise QualityLedgerError(f"quality obligation already closed: {key}")
                raise QualityLedgerError(f"unknown quality obligation: {key}")
            marker = f"NOT_ELIGIBLE_NO_FINAL_PREDICTION:{reason}"
            self._closed[key] = marker
            self.not_eligible += 1
            self._append(
                {
                    **base,
                    "capture_id": f"{base['stream_id']}:{int(base['frame_id'])}",
                    "event": "NOT_ELIGIBLE_NO_FINAL_PREDICTION",
                    "failure_reason": str(reason),
                    "quality_ticket_terminal": True,
                    "map_terminal": False,
                    "duplicate_identical": False,
                    "eligibility": "NOT_ELIGIBLE_NO_FINAL_PREDICTION",
                }
            )

    def pending_records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(value) for value in self._pending.values())

    def _append(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({name: row.get(name, "") for name in FIELDS})
        self._handle.flush()

    def summary(self) -> dict[str, int]:
        with self._lock:
            return {
                "registered": len(self._pending) + len(self._closed),
                "completed": len(self._closed),
                "pending": len(self._pending),
                "not_eligible_no_final_prediction": self.not_eligible,
                "quality_ack_outcomes": len(self._closed) - self.not_eligible,
                "identical_duplicates": self.identical_duplicates,
            }

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()

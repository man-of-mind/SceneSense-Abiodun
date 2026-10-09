"""UE-side terminal ledger for the direct edge-to-map architecture.

One transmission obligation is created the instant a capture commits to the
radio, and it ends in exactly one terminal outcome:

* a map feedback terminal (``RESULT_INSTALLED``, ``SUPERSEDED_PENDING``,
  ``STALE_BEFORE_MAP``, ``MAP_REJECTED``),
* an edge terminal control (``STALE_BEFORE_EDGE``, ``SUPERSEDED_PENDING``,
  ``STALE_BEFORE_MAP``), or
* the UE-local ``FEEDBACK_TIMEOUT`` watchdog.

A message that arrives after the obligation already closed is written as an
explicitly non-terminal diagnostic row; it never replaces the terminal and never
increments the terminal count. That is what makes "exactly one terminal per
obligation" checkable from the CSV alone.

Physical map freshness ends at ``install_timestamp``. ``feedback_received_at``
is recorded in the same row but is a separate controller-observation delay and
is never used as an installation time.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from . import protocol
from .protocol import (
    DIRECT_MAP_FEEDBACK_SCHEMA,
    EDGE_TERMINAL_CONTROL_SCHEMA,
    OUTCOME_FEEDBACK_TIMEOUT,
    OUTCOME_RESULT_INSTALLED,
)


FIELDS = (
    "experiment_id",
    "cell_id",
    "stream_id",
    "capture_id",
    "frame_id",
    "capture_at",
    "action_id",
    "profile_id",
    "service_deadline_at",
    "ack_timeout_at",
    "terminal_source",
    "outcome",
    "agent_credit",
    "status",
    "result_status",
    "install_timestamp",
    "map_ingest_at",
    "feedback_emit_at",
    "feedback_received_at",
    "map_age_at_install_ms",
    "feedback_observation_delay_ms",
    "direct_update_bytes",
    "direct_update_datagrams",
    "feedback_bytes",
    "superseded_by_frame_id",
    "terminal",
    "accepted",
    "late",
    "timeout_seen",
    "rejection_reason",
)

SOURCE_MAP = "EDGE_SPATIAL_MAP"
SOURCE_EDGE = "EDGE_INFERENCE_SERVICE"
SOURCE_UE = "UE_LOCAL_WATCHDOG"


class DirectLedgerError(RuntimeError):
    """A terminal message violated the direct-architecture ledger contract."""


class DirectTerminalLedger:
    """Create-only CSV evidence sink with exactly-one-terminal enforcement."""

    def __init__(
        self,
        *,
        output_csv: Path,
        experiment_id: str,
        cell_id: str,
    ) -> None:
        self.experiment_id = str(experiment_id)
        self.cell_id = str(cell_id)
        self.pending: dict[str, dict[str, Any]] = {}
        self.closed: dict[str, str] = {}
        self.timed_out: set[str] = set()
        self.late_messages = 0
        self.lock = threading.Lock()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.handle = output_csv.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(FIELDS))
        self.writer.writeheader()
        self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()

    # -- obligations -------------------------------------------------------

    def register_capture(
        self,
        *,
        stream_id: str,
        capture_id: str,
        frame_id: int,
        capture_at: float,
        action_id: str,
        profile_id: str,
        service_deadline_at: float,
        ack_timeout_at: float,
    ) -> None:
        record = {
            "stream_id": str(stream_id),
            "capture_id": str(capture_id),
            "frame_id": int(frame_id),
            "capture_at": float(capture_at),
            "action_id": str(action_id),
            "profile_id": str(profile_id),
            "service_deadline_at": float(service_deadline_at),
            "ack_timeout_at": float(ack_timeout_at),
        }
        with self.lock:
            if record["capture_id"] in self.pending or record["capture_id"] in self.closed:
                raise DirectLedgerError(
                    f"duplicate capture registration: {capture_id}"
                )
            self.pending[record["capture_id"]] = record

    def _append(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow({field: row.get(field, "") for field in FIELDS})
        self.handle.flush()

    # -- terminals ---------------------------------------------------------

    def record_message(self, message: Mapping[str, Any], received_at: float) -> dict[str, Any]:
        """Record one map feedback or edge terminal control message."""

        schema = str(message.get("schema") or "")
        if schema == DIRECT_MAP_FEEDBACK_SCHEMA:
            protocol.validate_map_feedback(message)
            source = SOURCE_MAP
        elif schema == EDGE_TERMINAL_CONTROL_SCHEMA:
            protocol.validate_edge_terminal_control(message)
            source = SOURCE_EDGE
        else:
            raise DirectLedgerError(f"unknown control schema: {schema!r}")
        protocol.assert_no_object_records(message)

        capture_id = str(message.get("capture_id") or "")
        with self.lock:
            base = self.pending.get(capture_id)
            already = self.closed.get(capture_id)
            if base is None and already is None:
                raise DirectLedgerError(f"feedback for unknown capture: {capture_id}")
            if base is None:
                # Obligation already terminal: a late or duplicate arrival is a
                # diagnostic row only. It must not become a second terminal.
                self.late_messages += 1
                row = self._row(
                    message,
                    source=source,
                    base={
                        "stream_id": str(message.get("stream_id") or ""),
                        "capture_id": capture_id,
                        "frame_id": int(message.get("frame_id", 0)),
                        "capture_at": float(message.get("capture_timestamp") or 0.0),
                        "action_id": str(message.get("action_id", "")),
                        "profile_id": str(message.get("profile_id", "")),
                        "service_deadline_at": float(
                            message.get("service_deadline_at") or 0.0
                        ),
                        "ack_timeout_at": float(message.get("ack_timeout_at") or 0.0),
                    },
                    received_at=received_at,
                    terminal=False,
                    timeout_seen=capture_id in self.timed_out,
                )
                self._append(row)
                return row

            self._verify_identity(message, base)
            terminal = bool(message.get("terminal"))
            row = self._row(
                message,
                source=source,
                base=base,
                received_at=received_at,
                terminal=terminal,
                timeout_seen=capture_id in self.timed_out,
            )
            self._append(row)
            if terminal:
                self.pending.pop(capture_id, None)
                self.closed[capture_id] = str(message.get("outcome") or "")
            return row

    def _verify_identity(
        self, message: Mapping[str, Any], base: Mapping[str, Any]
    ) -> None:
        if int(message["frame_id"]) != int(base["frame_id"]):
            raise DirectLedgerError("terminal frame identity mismatch")
        if str(message["action_id"]) != str(base["action_id"]):
            raise DirectLedgerError("terminal action identity mismatch")
        if str(message.get("profile_id", base["profile_id"])) != str(base["profile_id"]):
            raise DirectLedgerError("terminal profile identity mismatch")
        if str(message.get("stream_id") or "") != str(base["stream_id"]):
            raise DirectLedgerError("terminal stream identity mismatch")
        if str(message.get("outcome") or "") == OUTCOME_RESULT_INSTALLED:
            if message.get("install_timestamp") in (None, ""):
                raise DirectLedgerError("RESULT_INSTALLED lacks install_timestamp")

    def _row(
        self,
        message: Mapping[str, Any],
        *,
        source: str,
        base: Mapping[str, Any],
        received_at: float,
        terminal: bool,
        timeout_seen: bool,
    ) -> dict[str, Any]:
        outcome = str(message.get("outcome") or "")
        install_at = message.get("install_timestamp", "")
        emit_at = message.get("feedback_emit_at", message.get("emit_at", ""))
        late = bool(timeout_seen)
        if install_at not in (None, ""):
            late = late or float(install_at) > float(base["service_deadline_at"])
        observation_delay_ms = ""
        if emit_at not in (None, ""):
            observation_delay_ms = (float(received_at) - float(emit_at)) * 1000.0
        accepted = outcome == OUTCOME_RESULT_INSTALLED
        return {
            "experiment_id": self.experiment_id,
            "cell_id": self.cell_id,
            **base,
            "terminal_source": source,
            "outcome": outcome,
            "agent_credit": str(message.get("agent_credit") or ""),
            # Legacy-compatible status columns so the registered output schema
            # and every downstream consumer keep working unchanged.
            "status": "ACK_INSTALLED" if accepted else "NACK_REJECTED",
            "result_status": (
                "DECODED_RESULT_ACCEPTED_AND_INSTALLED"
                if accepted
                else "RESULT_REJECTED"
            ),
            "install_timestamp": install_at,
            "map_ingest_at": message.get("map_ingest_at", ""),
            "feedback_emit_at": emit_at,
            "feedback_received_at": received_at,
            "map_age_at_install_ms": message.get("map_age_at_install_ms", ""),
            "feedback_observation_delay_ms": observation_delay_ms,
            "direct_update_bytes": message.get("direct_update_bytes", ""),
            "direct_update_datagrams": message.get("direct_update_datagrams", ""),
            "feedback_bytes": message.get("_feedback_bytes", ""),
            "superseded_by_frame_id": message.get("superseded_by_frame_id", ""),
            "terminal": bool(terminal),
            "accepted": outcome == OUTCOME_RESULT_INSTALLED,
            "late": late,
            "timeout_seen": timeout_seen,
            "rejection_reason": str(message.get("rejection_reason") or ""),
        }

    def record_expired(self, now: float | None = None) -> int:
        """Close obligations whose ACK horizon elapsed. No resend is performed."""

        observed = time.time() if now is None else float(now)
        count = 0
        with self.lock:
            for capture_id, base in list(self.pending.items()):
                if observed < float(base["ack_timeout_at"]):
                    continue
                self.timed_out.add(capture_id)
                self._append(
                    {
                        "experiment_id": self.experiment_id,
                        "cell_id": self.cell_id,
                        **base,
                        "terminal_source": SOURCE_UE,
                        "outcome": OUTCOME_FEEDBACK_TIMEOUT,
                        "agent_credit": protocol.classify_agent_credit(
                            OUTCOME_FEEDBACK_TIMEOUT
                        ),
                        "status": "TIMEOUT_NO_ACK",
                        "result_status": "NO_AUTHORITATIVE_FEEDBACK",
                        "feedback_received_at": observed,
                        "terminal": True,
                        "accepted": False,
                        "late": True,
                        "timeout_seen": True,
                        "rejection_reason": "ACK_TIMEOUT_EXPIRED_NO_RESEND",
                    }
                )
                self.pending.pop(capture_id, None)
                self.closed[capture_id] = OUTCOME_FEEDBACK_TIMEOUT
                count += 1
        return count

    def summary(self) -> dict[str, Any]:
        with self.lock:
            outcomes: dict[str, int] = {}
            for value in self.closed.values():
                outcomes[value] = outcomes.get(value, 0) + 1
            return {
                "obligations_closed": len(self.closed),
                "obligations_open": len(self.pending),
                "late_nonterminal_messages": self.late_messages,
                "terminal_outcomes": dict(sorted(outcomes.items())),
            }

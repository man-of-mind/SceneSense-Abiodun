#!/usr/bin/env python3
"""Minimal UE-side map-install feedback ledger for one campaign cell."""

from __future__ import annotations

import csv
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping


FIELDS = (
    "experiment_id",
    "cell_id",
    "stream_id",
    "capture_id",
    "frame_id",
    "capture_at",
    "action_id",
    "service_deadline_at",
    "edge_install_at",
    "install_timestamp",
    "feedback_emit_at",
    "feedback_received_at",
    "ack_timeout_at",
    "result_status",
    "status",
    "accepted",
    "late",
    "timeout_seen",
    "rejection_reason",
)
REMOTE_STATUSES = {"ACK_INSTALLED", "NACK_REJECTED", "NACK_REASSEMBLY_TIMEOUT"}


class FeedbackContractError(RuntimeError):
    """An install-feedback message violates the frozen cell contract."""


class InstallFeedbackLedger:
    """Non-blocking capture tracker plus create-only CSV evidence sink.

    The capture producer calls :meth:`register_capture` and continues.  A
    receiver thread calls :meth:`receive_once`; a watchdog calls
    :meth:`record_expired`.  No resend is performed.  An ACK arriving after a
    timeout produces a second, explicitly late row and never erases the
    timeout violation.
    """

    def __init__(
        self,
        *,
        output_csv: Path,
        experiment_id: str,
        cell_id: str,
        bind_host: str,
        bind_port: int,
    ) -> None:
        self.experiment_id = str(experiment_id)
        self.cell_id = str(cell_id)
        self.pending: dict[str, dict[str, Any]] = {}
        self.timed_out: set[str] = set()
        self.lock = threading.Lock()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.handle = output_csv.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(FIELDS))
        self.writer.writeheader()
        self.handle.flush()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((str(bind_host), int(bind_port)))
        self.socket.settimeout(0.05)

    def close(self) -> None:
        try:
            self.socket.close()
        finally:
            self.handle.flush()
            self.handle.close()

    def register_capture(
        self,
        *,
        stream_id: str,
        capture_id: str,
        frame_id: int,
        capture_at: float,
        action_id: str,
        service_deadline_at: float,
        ack_timeout_at: float,
    ) -> None:
        record = {
            "stream_id": str(stream_id),
            "capture_id": str(capture_id),
            "frame_id": int(frame_id),
            "capture_at": float(capture_at),
            "action_id": str(action_id),
            "service_deadline_at": float(service_deadline_at),
            "ack_timeout_at": float(ack_timeout_at),
        }
        with self.lock:
            if record["capture_id"] in self.pending:
                raise FeedbackContractError(f"duplicate capture registration: {capture_id}")
            self.pending[record["capture_id"]] = record

    def _append(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow({field: row.get(field, "") for field in FIELDS})
        self.handle.flush()

    def receive_once(self) -> bool:
        try:
            packet, _source = self.socket.recvfrom(65535)
        except socket.timeout:
            return False
        received_at = time.time()
        try:
            message = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FeedbackContractError(f"invalid feedback datagram: {exc}") from exc
        if not isinstance(message, dict) or message.get("schema") != "scenesense.map_install_feedback.v1":
            raise FeedbackContractError("feedback schema mismatch")
        status = str(message.get("status") or "")
        if status not in REMOTE_STATUSES:
            raise FeedbackContractError(f"invalid remote feedback status: {status}")
        capture_id = str(message.get("capture_id") or "")
        with self.lock:
            base = self.pending.get(capture_id)
            if base is None:
                raise FeedbackContractError(f"feedback for unknown capture: {capture_id}")
            timeout_seen = capture_id in self.timed_out
            install_at = message.get("install_timestamp", "")
            if status == "ACK_INSTALLED":
                required = (message.get("frame_id"), message.get("capture_timestamp"), message.get("action_id"), install_at, message.get("result_status"))
                if any(value in (None, "") for value in required):
                    raise FeedbackContractError("ACK_INSTALLED lacks frame/capture/action/install/result fields")
                if int(message["frame_id"]) != int(base["frame_id"]):
                    raise FeedbackContractError("ACK_INSTALLED frame mismatch")
                if str(message["action_id"]) != str(base["action_id"]):
                    raise FeedbackContractError("ACK_INSTALLED action mismatch")
                if str(message["result_status"]) != "DECODED_RESULT_ACCEPTED_AND_INSTALLED":
                    raise FeedbackContractError("ACK_INSTALLED result status is not authoritative")
            late = bool(timeout_seen)
            if install_at not in (None, ""):
                late = late or float(install_at) > float(base["service_deadline_at"])
            self._append(
                {
                    "experiment_id": self.experiment_id,
                    "cell_id": self.cell_id,
                    **base,
                    "edge_install_at": install_at,
                    "install_timestamp": install_at,
                    "feedback_emit_at": message.get("feedback_emit_at", ""),
                    "feedback_received_at": received_at,
                    "result_status": message.get("result_status", ""),
                    "status": status,
                    "accepted": status == "ACK_INSTALLED",
                    "late": late,
                    "timeout_seen": timeout_seen,
                    "rejection_reason": message.get("rejection_reason", ""),
                }
            )
            self.pending.pop(capture_id, None)
        return True

    def record_reassembly_failure(self, *, capture_id: str, reason: str) -> None:
        """Record an identifiable edge reassembly terminal without inference."""
        with self.lock:
            base = self.pending.pop(str(capture_id), None)
            if base is None:
                raise FeedbackContractError(f"reassembly failure for unknown capture: {capture_id}")
            self._append(
                {
                    "experiment_id": self.experiment_id,
                    "cell_id": self.cell_id,
                    **base,
                    "result_status": "FEATURE_REASSEMBLY_FAILED",
                    "status": "NACK_REASSEMBLY_TIMEOUT",
                    "accepted": False,
                    "late": False,
                    "timeout_seen": False,
                    "rejection_reason": str(reason),
                }
            )

    def record_expired(self, now: float | None = None) -> int:
        """Append UE-local TIMEOUT_NO_ACK rows; never resend or delete evidence."""
        observed = time.time() if now is None else float(now)
        count = 0
        with self.lock:
            for capture_id, base in list(self.pending.items()):
                if capture_id in self.timed_out or observed < float(base["ack_timeout_at"]):
                    continue
                self.timed_out.add(capture_id)
                self._append(
                    {
                        "experiment_id": self.experiment_id,
                        "cell_id": self.cell_id,
                        **base,
                        "feedback_received_at": observed,
                        "result_status": "NO_AUTHORITATIVE_FEEDBACK",
                        "status": "TIMEOUT_NO_ACK",
                        "accepted": False,
                        "late": True,
                        "timeout_seen": True,
                        "rejection_reason": "ACK_TIMEOUT_EXPIRED_NO_RESEND",
                    }
                )
                count += 1
        return count

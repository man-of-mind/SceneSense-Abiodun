"""Host-only UE ledgers that reconcile map/edge terminals to the SFD3 identity.

:class:`Run4TerminalLedgerV2` subclasses the unchanged ``DirectTerminalLedger``
and replaces only its identity check with the exact Run-4 identity;
:class:`Run4CompatLedgerV2` is the collector-facing ``CompatDirectLedger`` whose
registration pulls the identity the runtime staged for that capture.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as DA
from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP
from rl_agent.splitfusion_direct_edge_map_v1.ue_ledger import DirectTerminalLedger

from .run4_map_protocol_v2 import (
    RUN4_FEEDBACK_SCHEMA,
    RUN4_TERMINAL_SCHEMA,
    validate_run4_edge_terminal,
    validate_run4_identity,
    validate_run4_map_feedback,
)

__all__ = ["Run4TerminalLedgerV2", "Run4CompatLedgerV2"]

_require = DP._require




class Run4TerminalLedgerV2(DirectTerminalLedger):
    """DirectTerminalLedger whose identity check is the exact Run-4 identity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._run4_identity: dict[str, dict[str, Any]] = {}
        self._current: Optional[Mapping[str, Any]] = None
        self._message_lock = threading.Lock()

    def register_run4_capture(self, *, identity: Mapping[str, Any], **kwargs: Any) -> None:
        validate_run4_identity(identity)
        anchor = identity["anchor_action_id"]
        capture_id = str(kwargs["capture_id"])
        with self.lock:
            _require(capture_id not in self._run4_identity, "duplicate Run-4 capture")
            self._run4_identity[capture_id] = dict(identity)
        super().register_capture(
            action_id="" if anchor is None else str(anchor),
            profile_id=str(kwargs.pop("anchor_profile_id") or ""), **kwargs)

    def record_message(self, message: Mapping[str, Any], received_at: float) -> dict[str, Any]:
        schema = str(message.get("schema") or "")
        if schema == RUN4_FEEDBACK_SCHEMA:
            validate_run4_map_feedback(message)
            legacy_schema = DP.DIRECT_MAP_FEEDBACK_SCHEMA
        elif schema == RUN4_TERMINAL_SCHEMA:
            validate_run4_edge_terminal(message)
            legacy_schema = DP.EDGE_TERMINAL_CONTROL_SCHEMA
        else:
            raise DP.DirectMapProtocolError(f"not a Run-4 control schema: {schema!r}")
        capture_id = str(message.get("capture_id") or "")
        with self.lock:
            registered = self.pending.get(capture_id) or {}
        view = {**dict(message), "schema": legacy_schema,
                "action_id": registered.get("action_id", ""),
                "profile_id": registered.get("profile_id", "")}
        with self._message_lock:
            self._current = message
            try:
                return super().record_message(view, received_at)
            finally:
                self._current = None

    def _verify_identity(self, message: Mapping[str, Any], base: Mapping[str, Any]) -> None:
        super()._verify_identity(message, base)
        current = self._current
        expected = self._run4_identity.get(str(base["capture_id"]))
        _require(current is not None and expected is not None,
                 "Run-4 capture has no registered identity")
        _require(dict(current.get("run4_identity") or {}) == expected,
                 "map/edge terminal Run-4 identity differs from the transmitted SFD3 identity")


class Run4CompatLedgerV2(DA.CompatDirectLedger):
    """Collector-facing ledger; registration pulls the runtime-staged identity."""

    def __init__(self, ledger: Run4TerminalLedgerV2) -> None:
        super().__init__(ledger, profile_id="")
        self._staged: dict[str, dict[str, Any]] = {}
        self._staged_lock = threading.Lock()

    def stage(self, capture_id: str, *, identity: Mapping[str, Any],
              anchor_profile_id: Optional[str]) -> None:
        with self._staged_lock:
            _require(str(capture_id) not in self._staged, "capture already staged")
            self._staged[str(capture_id)] = {"identity": dict(identity),
                                             "anchor_profile_id": anchor_profile_id}

    def register_capture(self, *, stream_id: str, capture_id: str, frame_id: int,
                         capture_at: float, action_id: str,
                         service_deadline_at: float, ack_timeout_at: float) -> None:
        del action_id  # the collector's cell action is not this frame's action
        with self._staged_lock:
            staged = self._staged.pop(str(capture_id), None)
        _require(staged is not None, f"no staged Run-4 identity for {capture_id}")
        self._ledger.register_run4_capture(
            identity=staged["identity"], anchor_profile_id=staged["anchor_profile_id"],
            stream_id=stream_id, capture_id=capture_id, frame_id=frame_id,
            capture_at=capture_at, service_deadline_at=service_deadline_at,
            ack_timeout_at=ack_timeout_at)
        with self._view_lock:
            self._pending[str(capture_id)] = {"frame_id": int(frame_id)}

"""Versioned Run-4 direct-map protocol, map-process proxy and UE ledger.

The legacy direct-map messages identify the action only by a catalog
``action_id`` in ``[0, 72)``.  Run-4 frames may be off-anchor, so this module
adds three *new* schemas carrying the exact continuous identity
(:func:`run4_identity`) in place of ``action_id``/``profile_id``:

* ``splitfusion_direct_object_map_update.run4.v1``  (edge -> map)
* ``splitfusion_direct_map_feedback.run4.v1``       (map -> UE)
* ``splitfusion_edge_terminal_control.run4.v1``     (edge -> UE)

Every legacy structural rule is kept (finite records, record count, deadline
order, no object records toward the UE, registered outcomes and agent
credits).  Nothing legacy is edited: :class:`MapProtocolProxyV2` is installed
**only inside the Run-4 map-server process** and forwards every legacy call
unchanged.
No off-anchor q is ever mapped to a catalog ``action_id``/``profile_id``.
This module is edge-safe (no host-only imports); the UE ledgers live in
``run4_ue_ledger_v2``.
"""

from __future__ import annotations

import threading
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP

from . import continuous_execution_v2 as X
from .phase6_decision_engine_v2 import FALLBACK_SEQ

__all__ = [
    "RUN4_UPDATE_SCHEMA",
    "RUN4_FEEDBACK_SCHEMA",
    "RUN4_TERMINAL_SCHEMA",
    "IDENTITY_FIELDS",
    "frame_kind",
    "run4_identity",
    "validate_run4_identity",
    "build_run4_map_update",
    "validate_run4_map_update",
    "build_run4_map_feedback",
    "validate_run4_map_feedback",
    "build_run4_edge_terminal",
    "validate_run4_edge_terminal",
    "MapProtocolProxyV2",
    "run4_install_document",
    "gt_identity",
]

RUN4_UPDATE_SCHEMA = "splitfusion_direct_object_map_update.run4.v1"
RUN4_FEEDBACK_SCHEMA = "splitfusion_direct_map_feedback.run4.v1"
RUN4_TERMINAL_SCHEMA = "splitfusion_edge_terminal_control.run4.v1"
PROTOCOL_VERSION = 1
IDENTITY_FIELDS = (
    "session_uuid", "controller_lineage_sha256", "decision_seq", "ticket_seq",
    "frame_id", "tensor_seq", "mode_id", "q_e4", "execution_bundle_sha256",
    "anchor_action_id", "reward_requested", "frame_kind",
)
FRAME_KINDS = ("POLICY_DECISION", "POLICY_HOLD", "FALLBACK")

_require = DP._require


def frame_kind(envelope: X.ExecutionEnvelopeV3) -> str:
    if envelope.decision_seq == FALLBACK_SEQ:
        DP._require(envelope.ticket_seq == FALLBACK_SEQ and not envelope.reward_requested,
                    "fallback frame must carry no ticket and no reward request")
        return "FALLBACK"
    return "POLICY_DECISION" if envelope.reward_requested else "POLICY_HOLD"


def run4_identity(envelope: X.ExecutionEnvelopeV3) -> dict[str, Any]:
    """The exact SFD3 action/ticket identity carried downstream."""
    return {
        "session_uuid": envelope.session_uuid,
        "controller_lineage_sha256": envelope.controller_lineage_sha256,
        "decision_seq": int(envelope.decision_seq),
        "ticket_seq": int(envelope.ticket_seq),
        "frame_id": int(envelope.frame_id),
        "tensor_seq": int(envelope.tensor_seq),
        "mode_id": int(envelope.mode_id),
        "q_e4": int(envelope.q_e4),
        "execution_bundle_sha256": envelope.execution_bundle_sha256,
        "anchor_action_id": envelope.anchor_action_id,
        "reward_requested": bool(envelope.reward_requested),
        "frame_kind": frame_kind(envelope),
    }


def validate_run4_identity(identity: Any, *, contract: Any = None) -> None:
    _require(isinstance(identity, Mapping), "run4_identity must be an object")
    _require(tuple(sorted(identity)) == tuple(sorted(IDENTITY_FIELDS)),
             "run4_identity field set drift")
    for name in ("decision_seq", "ticket_seq", "frame_id", "tensor_seq", "mode_id", "q_e4"):
        _require(type(identity[name]) is int and identity[name] >= 0,
                 f"run4_identity.{name} must be a non-negative int")
    _require(0 <= identity["mode_id"] < 12 and identity["q_e4"] <= 9800,
             "run4_identity mode/q outside the registered range")
    _require(type(identity["reward_requested"]) is bool, "reward_requested must be bool")
    anchor = identity["anchor_action_id"]
    _require(anchor is None or (type(anchor) is int and 0 <= anchor < 72),
             "anchor_action_id must be null or a registered anchor")
    _require(identity["frame_kind"] in FRAME_KINDS, "unknown frame kind")
    for name in ("controller_lineage_sha256", "execution_bundle_sha256"):
        value = identity[name]
        _require(isinstance(value, str) and len(value) == 64
                 and all(c in "0123456789abcdef" for c in value), f"{name} is not a digest")
    if identity["frame_kind"] == "FALLBACK":
        _require(identity["decision_seq"] == FALLBACK_SEQ
                 and identity["ticket_seq"] == FALLBACK_SEQ
                 and identity["reward_requested"] is False,
                 "fallback identity must carry no ticket or reward request")
    else:
        _require(identity["decision_seq"] != FALLBACK_SEQ, "policy frame uses sentinel")
        _require(identity["reward_requested"] == (identity["frame_kind"] == "POLICY_DECISION"),
                 "reward_requested disagrees with the frame kind")
    if contract is not None:
        profile = contract.resolve_q_e4(identity["mode_id"], identity["q_e4"])
        _require(profile.execution_bundle_sha256 == identity["execution_bundle_sha256"],
                 "execution bundle disagrees with the contract resolution")
        _require(profile.action_id == anchor,
                 "anchor identity disagrees with the contract resolution")


def gt_identity(*, run_id: str, cell_id: str, stream_id: str, frame_id: int,
                anchor_action_id: Optional[int], anchor_profile_id: Optional[str],
                capture_timestamp_ns: int) -> dict[str, Any]:
    """GT-evidence identity: real anchor identity or null, never fabricated."""
    return {"run_id": str(run_id), "cell_id": str(cell_id),
            "stream_id": str(stream_id), "frame_id": int(frame_id),
            "action_id": anchor_action_id, "profile_id": anchor_profile_id,
            "capture_timestamp_ns": int(capture_timestamp_ns)}


# ---------------------------------------------------------------------------
# Map update / feedback / edge terminal
# ---------------------------------------------------------------------------


def build_run4_map_update(
    *, run_id: str, cell_id: str, stream_id: str, frame_id: int, sequence_id: int,
    identity: Mapping[str, Any], decoder_identity: str, capture_timestamp_ns: int,
    records: Sequence[Mapping[str, Any]], service_deadline_at: float,
    ack_timeout_at: float, edge_timing: Mapping[str, Any],
    segmentation: Mapping[str, Any],
) -> dict[str, Any]:
    record_list = [dict(item) for item in records]
    scalars = DP.assert_finite_tree(record_list, path="$.records")
    document = {
        "schema": RUN4_UPDATE_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        "run_id": str(run_id), "cell_id": str(cell_id), "stream_id": str(stream_id),
        "frame_id": int(frame_id), "sequence_id": int(sequence_id),
        "run4_identity": dict(identity), "decoder_identity": str(decoder_identity),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "capture_timestamp": int(capture_timestamp_ns) / 1_000_000_000.0,
        "carla_timestamp": 0.0,
        "service_deadline_at": float(service_deadline_at),
        "ack_timeout_at": float(ack_timeout_at),
        "record_count": len(record_list), "record_scalar_count": int(scalars),
        "records": record_list, "edge_timing": dict(edge_timing),
        "segmentation": dict(segmentation),
    }
    validate_run4_map_update(document)
    return document


def validate_run4_map_update(document: Mapping[str, Any], *, contract: Any = None) -> None:
    _require(document.get("schema") == RUN4_UPDATE_SCHEMA, "Run-4 update schema drift")
    _require(int(document.get("protocol_version", 0)) == PROTOCOL_VERSION,
             "Run-4 update protocol version drift")
    for field in ("run_id", "cell_id", "stream_id", "frame_id", "capture_timestamp_ns",
                  "run4_identity"):
        _require(document.get(field) not in (None, ""), f"Run-4 update lacks {field}")
    _require(not {"action_id", "profile_id"} & set(document),
             "Run-4 update must not carry a catalog action_id/profile_id")
    identity = document["run4_identity"]
    validate_run4_identity(identity, contract=contract)
    _require(int(document["frame_id"]) == identity["frame_id"] >= 0,
             "Run-4 update frame identity drift")
    _require(int(document["sequence_id"]) == identity["tensor_seq"],
             "Run-4 update sequence must equal the tensor sequence")
    _require(int(document["capture_timestamp_ns"]) > 0, "capture timestamp must be positive")
    records = document.get("records")
    _require(isinstance(records, list), "records must be a list")
    _require(int(document.get("record_count", -1)) == len(records), "record_count drift")
    DP.assert_finite_tree(records, path="$.records")
    _require(float(document["ack_timeout_at"]) > float(document["service_deadline_at"]),
             "ACK timeout must exceed the service deadline")


def build_run4_map_feedback(*, update: Mapping[str, Any], outcome: str, terminal: bool,
                            install_timestamp: Optional[float], map_ingest_at: float,
                            feedback_emit_at: float,
                            map_age_at_install_ms: Optional[float],
                            rejection_reason: str = "", direct_update_bytes: int = 0,
                            direct_update_datagrams: int = 0,
                            superseded_by_frame_id: Optional[int] = None) -> dict[str, Any]:
    _require(outcome in DP.MAP_OUTCOMES, f"invalid map feedback outcome: {outcome!r}")
    _require((install_timestamp is not None) == (outcome == DP.OUTCOME_RESULT_INSTALLED),
             "only RESULT_INSTALLED feedback carries an install timestamp")
    credit = DP.classify_agent_credit(
        outcome, install_timestamp=install_timestamp,
        service_deadline_at=float(update["service_deadline_at"]))
    identity = update.get("run4_identity") if isinstance(update, Mapping) else None
    document = {
        "schema": RUN4_FEEDBACK_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        "source": "EDGE_SPATIAL_MAP", "run_id": str(update["run_id"]),
        "cell_id": str(update["cell_id"]), "stream_id": str(update["stream_id"]),
        "frame_id": int(update["frame_id"]),
        "capture_id": f"{update['stream_id']}:{int(update['frame_id'])}",
        "run4_identity": dict(identity) if isinstance(identity, Mapping) else None,
        "capture_timestamp_ns": int(update["capture_timestamp_ns"]),
        "capture_timestamp": int(update["capture_timestamp_ns"]) / 1_000_000_000.0,
        "service_deadline_at": float(update["service_deadline_at"]),
        "ack_timeout_at": float(update["ack_timeout_at"]),
        "outcome": str(outcome), "agent_credit": credit, "terminal": bool(terminal),
        "accepted": outcome == DP.OUTCOME_RESULT_INSTALLED,
        "install_timestamp": "" if install_timestamp is None else float(install_timestamp),
        "map_ingest_at": float(map_ingest_at), "feedback_emit_at": float(feedback_emit_at),
        "map_age_at_install_ms": ("" if map_age_at_install_ms is None
                                  else float(map_age_at_install_ms)),
        "rejection_reason": str(rejection_reason),
        "direct_update_bytes": int(direct_update_bytes),
        "direct_update_datagrams": int(direct_update_datagrams),
        "superseded_by_frame_id": ("" if superseded_by_frame_id is None
                                   else int(superseded_by_frame_id)),
        "edge_timing": dict(update.get("edge_timing") or {}),
    }
    DP.assert_no_object_records(document)
    validate_run4_map_feedback(document)
    return document


def validate_run4_map_feedback(document: Mapping[str, Any]) -> None:
    _require(document.get("schema") == RUN4_FEEDBACK_SCHEMA, "Run-4 feedback schema drift")
    _require(int(document.get("protocol_version", 0)) == PROTOCOL_VERSION,
             "Run-4 feedback version drift")
    _require(document.get("outcome") in DP.MAP_OUTCOMES, "Run-4 feedback outcome invalid")
    _require(document.get("agent_credit") in DP.AGENT_CREDITS, "agent credit invalid")
    _require(not {"action_id", "profile_id"} & set(document),
             "Run-4 feedback must not carry a catalog action_id/profile_id")
    if document["outcome"] == DP.OUTCOME_MAP_REJECTED and document.get("run4_identity") is None:
        pass  # an unparseable update is rejected without a trustworthy identity
    else:
        validate_run4_identity(document.get("run4_identity"))
        _require(document["run4_identity"]["frame_id"] == int(document["frame_id"]),
                 "Run-4 feedback frame identity drift")
    if document["outcome"] == DP.OUTCOME_RESULT_INSTALLED:
        _require(document.get("install_timestamp") not in (None, "")
                 and bool(document.get("accepted")), "installed feedback incomplete")
    DP.assert_no_object_records(document)


def build_run4_edge_terminal(*, run_id: str, cell_id: str, stream_id: str,
                             identity: Mapping[str, Any], capture_timestamp_ns: int,
                             service_deadline_at: float, ack_timeout_at: float,
                             outcome: str, stage: str, age_ms: float, emit_at: float,
                             superseded_by_frame_id: Optional[int] = None) -> dict[str, Any]:
    _require(outcome in DP.EDGE_CONTROL_OUTCOMES, f"invalid edge outcome: {outcome!r}")
    document = {
        "schema": RUN4_TERMINAL_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        "source": "EDGE_INFERENCE_SERVICE", "run_id": str(run_id), "cell_id": str(cell_id),
        "stream_id": str(stream_id), "frame_id": int(identity["frame_id"]),
        "capture_id": f"{stream_id}:{int(identity['frame_id'])}",
        "run4_identity": dict(identity),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "capture_timestamp": int(capture_timestamp_ns) / 1_000_000_000.0,
        "service_deadline_at": float(service_deadline_at),
        "ack_timeout_at": float(ack_timeout_at), "outcome": str(outcome),
        "agent_credit": DP.classify_agent_credit(outcome), "terminal": True,
        "accepted": False, "stage": str(stage), "age_ms": float(age_ms),
        "emit_at": float(emit_at),
        "superseded_by_frame_id": ("" if superseded_by_frame_id is None
                                   else int(superseded_by_frame_id)),
    }
    DP.assert_no_object_records(document)
    validate_run4_edge_terminal(document)
    return document


def validate_run4_edge_terminal(document: Mapping[str, Any]) -> None:
    _require(document.get("schema") == RUN4_TERMINAL_SCHEMA, "Run-4 terminal schema drift")
    _require(document.get("outcome") in DP.EDGE_CONTROL_OUTCOMES, "terminal outcome invalid")
    _require(document.get("agent_credit") in DP.AGENT_CREDITS, "agent credit invalid")
    _require(bool(document.get("terminal")), "edge terminal must be terminal")
    _require(not {"action_id", "profile_id"} & set(document),
             "Run-4 terminal must not carry a catalog action_id/profile_id")
    validate_run4_identity(document.get("run4_identity"))
    DP.assert_no_object_records(document)


# ---------------------------------------------------------------------------
# Map-process proxy (installed only in the Run-4 map server process)
# ---------------------------------------------------------------------------


class MapProtocolProxyV2:
    """``protocol`` stand-in for ``map_ingest``: Run-4 aware, legacy unchanged."""

    def __init__(self, legacy: ModuleType, *, contract: Any = None) -> None:
        self._legacy = legacy
        self._contract = contract
        self.run4_updates_validated = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._legacy, name)

    def validate_object_map_update(self, document: Mapping[str, Any]) -> None:
        if document.get("schema") == RUN4_UPDATE_SCHEMA:
            validate_run4_map_update(document, contract=self._contract)
            self.run4_updates_validated += 1
            return
        self._legacy.validate_object_map_update(document)

    def build_map_feedback(self, *, update: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        if isinstance(update, Mapping) and update.get("schema") == RUN4_UPDATE_SCHEMA:
            identity = update.get("run4_identity")
            try:
                validate_run4_identity(identity)
            except DP.DirectMapProtocolError:
                update = {**dict(update), "run4_identity": None}
            return build_run4_map_feedback(update=update, **kwargs)
        return self._legacy.build_map_feedback(update=update, **kwargs)


def run4_install_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Map-state view for the frozen baseline install: real anchor id or ''."""
    if document.get("schema") != RUN4_UPDATE_SCHEMA:
        return dict(document)
    anchor = document["run4_identity"]["anchor_action_id"]
    return {**dict(document), "action_id": "" if anchor is None else str(anchor)}

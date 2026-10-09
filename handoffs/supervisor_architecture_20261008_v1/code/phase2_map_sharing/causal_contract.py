"""Fail-closed causal state, audit logging, and counterfactual-arm isolation.

This module is deliberately independent of CARLA and OAI.  It defines the
runtime boundary that a future pilot collector must call before exposing a
field to either decision stage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from .schemas_v2 import (
    PLACEMENT_ACTIONS,
    PUBLICATION_ACTIONS,
    _reject_forbidden_runtime_keys,
)


CAUSAL_AUDIT_SCHEMA = "scenesense.causal_decision_audit.v1"
DECISION_STAGES = frozenset({"placement", "publication"})
FORBIDDEN_RUNTIME_SOURCES = frozenset({"evaluation_truth", "shadow_inference"})

PLACEMENT_FIELD_ALLOWLIST = frozenset(
    {
        "lagged_capacity_estimate_mbps",
        "capacity_estimate_sigma_mbps",
        "previous_action",
        "previous_delivery_success",
        "previous_latency_s",
        "previous_loss_fraction",
        "scheduler_credit",
        "in_flight_summary",
        "installed_map_summary",
        "prior_source_track_summary",
        "helper_state",
        "recipient_state",
        "recipient_state_message",
        "local_compute_headroom",
    }
)

PUBLICATION_FIELD_ALLOWLIST = frozenset(
    {
        "current_inference_result",
        "current_causal_tracks",
        "recipient_state_message",
        "scheduler_credit",
        "in_flight_summary",
        "installed_map_summary",
        "local_compute_headroom",
    }
)

FIELD_SOURCE_ALLOWLIST = {
    "lagged_capacity_estimate_mbps": frozenset({"prior_network_estimator"}),
    "capacity_estimate_sigma_mbps": frozenset({"prior_network_estimator"}),
    "previous_action": frozenset({"prior_completed_event"}),
    "previous_delivery_success": frozenset({"prior_completed_event"}),
    "previous_latency_s": frozenset({"prior_completed_event"}),
    "previous_loss_fraction": frozenset({"prior_completed_event"}),
    "scheduler_credit": frozenset({"scheduler_state"}),
    "in_flight_summary": frozenset({"scheduler_state"}),
    "installed_map_summary": frozenset({"recipient_map"}),
    "prior_source_track_summary": frozenset({"causal_tracker"}),
    "helper_state": frozenset({"helper_localization"}),
    "recipient_state": frozenset({"recipient_localization"}),
    "recipient_state_message": frozenset({"recipient_state_transport"}),
    "local_compute_headroom": frozenset({"local_compute_monitor"}),
    "current_inference_result": frozenset({"selected_inference"}),
    "current_causal_tracks": frozenset({"causal_tracker"}),
}


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required(name: str, value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def validate_runtime_payload(value: object) -> None:
    """Reject evaluation identity/truth recursively at the runtime boundary."""

    _reject_forbidden_runtime_keys(value, "runtime_state")
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime state must be finite and JSON-serializable") from exc


@dataclass(frozen=True)
class DecisionRecord:
    trajectory_id: str
    arm_id: str
    decision_id: str
    decision_stage: str
    decision_at_s: float
    clock_id: str
    action: str

    def validate(self) -> None:
        for name in ("trajectory_id", "arm_id", "decision_id", "clock_id", "action"):
            _required(name, getattr(self, name))
        if self.decision_stage not in DECISION_STAGES:
            raise ValueError("decision_stage must be placement or publication")
        allowed_actions = (
            PLACEMENT_ACTIONS if self.decision_stage == "placement" else PUBLICATION_ACTIONS
        )
        if self.action not in allowed_actions:
            raise ValueError(f"action is not valid for {self.decision_stage}")
        _finite("decision_at_s", self.decision_at_s)


@dataclass(frozen=True)
class CausalField:
    field_name: str
    value: object
    source_stage: str
    observed_at_s: float
    available_at_s: float
    consuming_decision_id: str
    consuming_decision_stage: str
    clock_id: str
    arm_id: str

    def validate_for(self, decision: DecisionRecord) -> None:
        decision.validate()
        for name in (
            "field_name",
            "source_stage",
            "consuming_decision_id",
            "consuming_decision_stage",
            "clock_id",
            "arm_id",
        ):
            _required(name, getattr(self, name))
        if self.source_stage in FORBIDDEN_RUNTIME_SOURCES:
            raise ValueError(f"{self.source_stage} cannot populate runtime policy state")
        if self.consuming_decision_id != decision.decision_id:
            raise ValueError("causal field references a different decision")
        if self.consuming_decision_stage != decision.decision_stage:
            raise ValueError("causal field references a different decision stage")
        if self.clock_id != decision.clock_id:
            raise ValueError("causal field and decision use different clock domains")
        if self.arm_id != decision.arm_id:
            raise ValueError("causal field crosses counterfactual arms")
        observed = _finite("observed_at_s", self.observed_at_s)
        available = _finite("available_at_s", self.available_at_s)
        if available + 1e-12 < observed:
            raise ValueError("field cannot be available before observation")
        if available > decision.decision_at_s + 1e-12:
            raise ValueError("field became available after the consuming decision")
        allowlist = (
            PLACEMENT_FIELD_ALLOWLIST
            if decision.decision_stage == "placement"
            else PUBLICATION_FIELD_ALLOWLIST
        )
        if self.field_name not in allowlist:
            raise ValueError(
                f"{self.field_name} is not allowlisted for {decision.decision_stage}"
            )
        if self.source_stage not in FIELD_SOURCE_ALLOWLIST[self.field_name]:
            raise ValueError(
                f"{self.source_stage} cannot produce runtime field {self.field_name}"
            )
        validate_runtime_payload(self.value)


@dataclass(frozen=True)
class CausalDecisionAudit:
    decision: DecisionRecord
    fields: Tuple[CausalField, ...]

    def validate(self) -> None:
        self.decision.validate()
        names = set()
        for field in self.fields:
            field.validate_for(self.decision)
            if field.field_name in names:
                raise ValueError("a causal decision cannot contain duplicate field names")
            names.add(field.field_name)

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema": CAUSAL_AUDIT_SCHEMA,
            "decision": asdict(self.decision),
            "fields": [asdict(field) for field in self.fields],
        }


class CausalAuditWriter:
    """Create-only JSONL writer; existing evidence is never overwritten."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        self.records_written = 0

    def write(self, audit: CausalDecisionAudit) -> str:
        payload = audit.to_dict()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        envelope = {**payload, "record_sha256": digest}
        self._stream.write(json.dumps(envelope, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()
        self.records_written += 1
        return digest

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "CausalAuditWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class ArmStateToken:
    trajectory_id: str
    arm_id: str
    revision: int


class CounterfactualArmRegistry:
    """Independent state stores for offline counterfactual policy arms."""

    def __init__(
        self,
        allowed_arm_ids: Sequence[str] = (
            "ego_only",
            "send_everything",
            "hazard_only",
        ),
    ) -> None:
        self.allowed_arm_ids = frozenset(_required("arm_id", item) for item in allowed_arm_ids)
        if not self.allowed_arm_ids:
            raise ValueError("at least one counterfactual arm must be allowed")
        self._states: Dict[tuple[str, str], object] = {}
        self._revisions: Dict[tuple[str, str], int] = {}

    def initialize(self, trajectory_id: str, arm_id: str, state: object) -> ArmStateToken:
        key = (_required("trajectory_id", trajectory_id), _required("arm_id", arm_id))
        if key[1] not in self.allowed_arm_ids:
            raise ValueError("counterfactual arm is not declared in the pilot contract")
        if key in self._states:
            raise ValueError("counterfactual arm is already initialized")
        validate_runtime_payload(state)
        self._states[key] = copy.deepcopy(state)
        self._revisions[key] = 0
        return ArmStateToken(key[0], key[1], 0)

    def read(self, token: ArmStateToken) -> object:
        self._validate_token(token)
        return copy.deepcopy(self._states[(token.trajectory_id, token.arm_id)])

    def commit(self, token: ArmStateToken, state: object) -> ArmStateToken:
        self._validate_token(token)
        validate_runtime_payload(state)
        key = (token.trajectory_id, token.arm_id)
        revision = token.revision + 1
        self._states[key] = copy.deepcopy(state)
        self._revisions[key] = revision
        return ArmStateToken(token.trajectory_id, token.arm_id, revision)

    def _validate_token(self, token: ArmStateToken) -> None:
        key = (token.trajectory_id, token.arm_id)
        if key not in self._states:
            raise ValueError("unknown trajectory/arm state token")
        if self._revisions[key] != token.revision:
            raise ValueError("stale or cross-branch arm state token")

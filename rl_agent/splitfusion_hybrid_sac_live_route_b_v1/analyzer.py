"""Read-only analyzer skeleton over a create-only pilot evidence directory.

The analyzer never writes into the session directory it reads.  It loads the
session header and the append-only frame stream, recomputes the accounting that
the later phases' acceptance gates depend on, and reports each gate as
``PASS`` / ``FAIL`` / ``NOT_EVALUABLE``.

``NOT_EVALUABLE`` is deliberate and is not a soft pass.  A Phase-1 evidence
directory contains no transport, quality or map fields, so the gates that read
them cannot be decided; saying so is the only honest verdict, and the summary
counts those gates separately from the ones that actually passed.  A gate is
never downgraded to make a run look qualified.

The gate inventory below is the Phase-1 skeleton.  Phases 4 through 7 extend it
with their own registered gates; they do not weaken these.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import pilot_contract as contract
from .evidence import PilotEvidenceWriter, validate_frame_record

__all__ = [
    "AnalysisError",
    "GATE_INVENTORY",
    "GateResult",
    "PilotAnalysisV1",
    "analyze_session",
    "load_session",
]

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"

#: ``gate id -> what it asserts``.  Phase 1 can decide the first four.
GATE_INVENTORY: Tuple[Tuple[str, str], ...] = (
    (
        "G01_SCHEMA_INTEGRITY",
        "every frame validates against the registered frame schema",
    ),
    (
        "G02_PILOT_BINDING",
        "every frame carries the same pilot label and contract digest as the "
        "session header",
    ),
    (
        "G03_FROZEN_ACTOR_IDENTITY",
        "every decision frame names one actor-state digest, and it is the "
        "pre-registered seed-17 update-10,000 one",
    ),
    (
        "G04_GENESIS_MATCHED_STATE",
        "no frame reports a non-zero previous-outcome feature",
    ),
    (
        "G05_ACTION_IDENTITY_CONSISTENCY",
        "an anchor action_id appears if and only if the measurement status is "
        "MEASURED_ANCHOR",
    ),
    (
        "G06_DECISION_CADENCE",
        "reward_requested is true exactly on frames that opened a decision, "
        "and every decision holds at least k_min tensors",
    ),
    (
        "G07_TERMINAL_ACCOUNTING",
        "every opened ticket reaches exactly one terminal class",
    ),
    (
        "G08_NO_CROSS_FRAME_REWARD",
        "no realized reward is attached to a frame other than its own "
        "reward-requested tensor",
    ),
    (
        "G09_METRIC_VALIDITY",
        "every class metric carries a status, and an absent class is null "
        "rather than zero",
    ),
    (
        "G10_MAP_PATH_SEPARATION",
        "map installation is recorded separately and is never a precondition "
        "of the reward ACK",
    ),
)

_PHASE1_DECIDABLE = frozenset(
    {
        "G01_SCHEMA_INTEGRITY",
        "G02_PILOT_BINDING",
        "G03_FROZEN_ACTOR_IDENTITY",
        "G04_GENESIS_MATCHED_STATE",
        "G05_ACTION_IDENTITY_CONSISTENCY",
    }
)


class AnalysisError(RuntimeError):
    """A pilot evidence directory is missing, malformed or unreadable."""


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    verdict: str
    detail: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {"detail": self.detail, "gate_id": self.gate_id, "verdict": self.verdict}


@dataclass(frozen=True, slots=True)
class PilotAnalysisV1:
    session: Mapping[str, Any]
    frame_count: int
    decision_count: int
    reuse_count: int
    executed_modes: Tuple[int, ...]
    executed_q_e4: Tuple[int, ...]
    anchor_frame_count: int
    off_anchor_frame_count: int
    out_of_support_frame_count: int
    out_of_support_feature_counts: Mapping[str, int]
    gates: Tuple[GateResult, ...]

    @property
    def passed(self) -> Tuple[str, ...]:
        return tuple(g.gate_id for g in self.gates if g.verdict == PASS)

    @property
    def failed(self) -> Tuple[str, ...]:
        return tuple(g.gate_id for g in self.gates if g.verdict == FAIL)

    @property
    def not_evaluable(self) -> Tuple[str, ...]:
        return tuple(g.gate_id for g in self.gates if g.verdict == NOT_EVALUABLE)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "anchor_frame_count": self.anchor_frame_count,
            "decision_count": self.decision_count,
            "executed_modes": list(self.executed_modes),
            "executed_q_e4": list(self.executed_q_e4),
            "failed_gates": list(self.failed),
            "frame_count": self.frame_count,
            "gates": [gate.to_canonical_dict() for gate in self.gates],
            "not_evaluable_gates": list(self.not_evaluable),
            "off_anchor_frame_count": self.off_anchor_frame_count,
            "out_of_support_feature_counts": dict(
                self.out_of_support_feature_counts
            ),
            "out_of_support_frame_count": self.out_of_support_frame_count,
            "passed_gates": list(self.passed),
            "pilot_label": contract.PILOT_LABEL,
            "record": contract.ANALYSIS_SCHEMA_ID,
            "reuse_count": self.reuse_count,
            "session_record": self.session.get("record"),
            "training_support_limitation": contract.TRAINING_SUPPORT_LIMITATION,
        }

    def canonical_sha256(self) -> str:
        return contract.canonical_sha256(self.to_canonical_dict())


def load_session(directory: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read the session header and every frame record, validating each."""
    base = Path(directory)
    if not base.is_dir():
        raise AnalysisError(f"evidence directory {base} does not exist")
    session_path = base / PilotEvidenceWriter.SESSION_FILENAME
    frames_path = base / PilotEvidenceWriter.FRAME_FILENAME
    if not session_path.is_file():
        raise AnalysisError(f"session header is missing at {session_path}")
    if not frames_path.is_file():
        raise AnalysisError(f"frame stream is missing at {frames_path}")
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError("session header is unreadable") from exc
    frames: List[Dict[str, Any]] = []
    with frames_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"frame {number} is not valid JSON") from exc
            frames.append(validate_frame_record(record))
    return session, frames


def _gate(gate_id: str, ok: Optional[bool], detail: str) -> GateResult:
    if ok is None:
        return GateResult(gate_id, NOT_EVALUABLE, detail)
    return GateResult(gate_id, PASS if ok else FAIL, detail)


def analyze_session(
    directory: Path, *, expected_actor_state_sha256: Optional[str] = None
) -> PilotAnalysisV1:
    """Evaluate the registered gate inventory over one evidence directory."""
    session, frames = load_session(directory)

    decisions = [row for row in frames if row.get("opened_decision") is True]
    reuses = [row for row in frames if row.get("reused_held_action") is True]
    modes = sorted({int(row["executed_mode_id"]) for row in frames if row.get("executed_mode_id") is not None})
    q_values = sorted({int(row["executed_q_e4"]) for row in frames if row.get("executed_q_e4") is not None})
    anchors = [row for row in frames if row.get("executed_action_id") is not None]
    off_anchor = [
        row
        for row in frames
        if row.get("executed_measurement_status") == "UNMEASURED_OFF_ANCHOR"
    ]

    support_counts: Dict[str, int] = {}
    out_of_support_frames = 0
    for row in frames:
        audit = row.get("support_audit")
        if not isinstance(audit, Mapping):
            continue
        findings = audit.get("out_of_support") or []
        if findings:
            out_of_support_frames += 1
        for item in findings:
            name = str(item.get("feature"))
            support_counts[name] = support_counts.get(name, 0) + 1

    gates: List[GateResult] = []

    gates.append(
        _gate(
            "G01_SCHEMA_INTEGRITY",
            True,
            f"{len(frames)} frames validated against "
            f"{contract.EVIDENCE_FRAME_SCHEMA_ID}",
        )
    )

    binding_ok = all(
        row["pilot_label"] == session.get("pilot_label")
        and row["pilot_contract_sha256"] == session.get("pilot_contract_sha256")
        for row in frames
    )
    gates.append(
        _gate(
            "G02_PILOT_BINDING",
            binding_ok if frames else None,
            "frame and session pilot bindings agree"
            if binding_ok
            else "at least one frame carries a different pilot binding",
        )
    )

    digests = {
        row["actor_state_sha256"]
        for row in frames
        if row.get("actor_state_sha256") is not None
    }
    if not digests:
        actor_ok: Optional[bool] = None
        actor_detail = "no frame recorded an actor-state digest"
    else:
        actor_ok = len(digests) == 1
        actor_detail = f"{len(digests)} distinct actor-state digest(s)"
        if actor_ok and expected_actor_state_sha256 is not None:
            actor_ok = next(iter(digests)) == expected_actor_state_sha256
            actor_detail += "; matches the pre-registered actor" if actor_ok else (
                "; differs from the pre-registered actor"
            )
    gates.append(_gate("G03_FROZEN_ACTOR_IDENTITY", actor_ok, actor_detail))

    violations = []
    for row in frames:
        values = row.get("policy_features")
        if not isinstance(values, Sequence) or len(values) != len(
            contract.TRAINING_SUPPORT
        ):
            continue
        named = dict(zip(_feature_order(), values))
        for name in contract.POLICY_CONTROLLED_ZERO_FEATURES:
            if float(named[name]) != 0.0:
                violations.append((row.get("tensor_seq"), name))
    any_features = any(
        isinstance(row.get("policy_features"), Sequence) for row in frames
    )
    gates.append(
        _gate(
            "G04_GENESIS_MATCHED_STATE",
            (not violations) if any_features else None,
            "no previous-outcome feature is non-zero"
            if not violations
            else f"{len(violations)} non-zero previous-outcome feature(s)",
        )
    )

    identity_bad = [
        row
        for row in frames
        if row.get("executed_measurement_status") is not None
        and (row.get("executed_action_id") is not None)
        != (row.get("executed_measurement_status") == "MEASURED_ANCHOR")
    ]
    any_identity = any(
        row.get("executed_measurement_status") is not None for row in frames
    )
    gates.append(
        _gate(
            "G05_ACTION_IDENTITY_CONSISTENCY",
            (not identity_bad) if any_identity else None,
            "anchor identity agrees with measurement status on every frame"
            if not identity_bad
            else f"{len(identity_bad)} frame(s) disagree",
        )
    )

    for gate_id, description in GATE_INVENTORY:
        if gate_id in _PHASE1_DECIDABLE:
            continue
        gates.append(
            _gate(
                gate_id,
                None,
                f"{description}: the required fields are not produced before "
                f"the phase that owns them",
            )
        )

    ordered = {gate.gate_id: gate for gate in gates}
    return PilotAnalysisV1(
        session=session,
        frame_count=len(frames),
        decision_count=len(decisions),
        reuse_count=len(reuses),
        executed_modes=tuple(modes),
        executed_q_e4=tuple(q_values),
        anchor_frame_count=len(anchors),
        off_anchor_frame_count=len(off_anchor),
        out_of_support_frame_count=out_of_support_frames,
        out_of_support_feature_counts=dict(sorted(support_counts.items())),
        gates=tuple(ordered[gate_id] for gate_id, _ in GATE_INVENTORY),
    )


def _feature_order() -> Tuple[str, ...]:
    """The registered order, imported lazily.

    The state contract pulls in the scene descriptors, and those pull in
    OpenCV.  A read-only analyzer should be runnable on a host that has the
    evidence but not the sensor stack, so the import happens only when a frame
    actually carries a feature vector to check.
    """
    from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
        POLICY_FEATURE_ORDER,
    )

    return POLICY_FEATURE_ORDER

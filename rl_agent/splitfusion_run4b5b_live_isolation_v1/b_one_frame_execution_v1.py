"""Exactly-one GT-free operational loop for split-host engineering.

This module deliberately does not call, wrap, patch, or relax the frozen
300-frame UE process.  It reuses its public transmission/receiver protocols
and the authoritative operational ACK and trace stores, while enforcing one
transmit, one policy decision and one terminal outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from . import b_ue_process_v1 as U
from . import operational_trace_v1 as T
from . import operational_ack_v1 as A


SCHEMA = "scenesense.splitfusion.run4b5b.one_frame_execution.v1"
RESULT_SCHEMA = "scenesense.splitfusion.run4b5b.one_frame_execution_result.v1"
REPORT_NAME = "ONE_FRAME_UE_REPORT.json"
RESULT_NAME = "ONE_FRAME_UE_RESULT.json"


class OneFrameExecutionError(RuntimeError):
    """The engineering-only one-frame loop failed closed."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise OneFrameExecutionError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise OneFrameExecutionError("value is not canonical JSON") from exc


def execute_one(
        request: U.BUEProcessRequestV1, pipeline: U.BUEPipelineV1,
        receiver: U.AckReceiverV1,
        ) -> dict[str, Any]:
    """Execute exactly one decision and seal GT-free operational evidence."""
    _require(type(request.transmitted_budget) is int
             and request.transmitted_budget == 1,
             "engineering request budget is not exactly one")
    _require(pipeline.variant is request.variant,
             "pipeline variant differs from request")
    _require(pipeline.feature_schema_sha256 == request.feature_schema_sha256,
             "pipeline feature schema differs")
    _require(pipeline.actor_boundary_sha256 == request.actor_boundary_sha256,
             "pipeline actor boundary differs")
    _require(not request.output_root.exists()
             and not request.evidence_root.exists(),
             "one-frame output/evidence roots must be create-only")
    request.output_root.mkdir(parents=True)
    request.evidence_root.mkdir(parents=True)
    operational = A.OperationalEvidenceStoreV1.create(
        request.evidence_root / "operational_evidence")
    traces = T.OperationalTraceStoreV1.create(
        request.evidence_root / "operational_trace")
    ledger = A.OperationalAckLedgerV1(evidence_store=operational)
    primary: Optional[BaseException] = None
    sent: Optional[U.BTransmissionV1] = None
    outcome: Optional[A.OperationalOutcomeV1] = None
    try:
        sent = pipeline.transmit_next(0, None)
        _require(type(sent) is U.BTransmissionV1,
                 "pipeline returned a foreign transmission")
        _require(sent.decision_frame is True,
                 "one-frame handshake did not transmit a policy decision")
        _require(sent.identity.run_id == request.run_id,
                 "transmission run identity drift")
        _require(sent.gt_objects is None and sent.gt_semantic_mask is None
                 and sent.gt_recorded_monotonic_raw_ns is None,
                 "live CARLA GT entered the operational one-frame loop")
        ledger.open(sent.identity, sent.action_open_monotonic_raw_ns)
        deadline = sent.action_open_monotonic_raw_ns + A.ACK_DEADLINE_NS
        received = receiver.receive_until(sent.identity, deadline)
        if received is None:
            ledger.poll(deadline + 1)
        else:
            packet, receipt = received
            ledger.receive(packet, receipt)
            if ledger.outcome(sent.identity) is None:
                ledger.poll(deadline + 1)
        outcome = ledger.outcome(sent.identity)
        _require(type(outcome) is A.OperationalOutcomeV1,
                 "one-frame decision did not reach a terminal")
        traces.write(outcome, sent.payload_bytes)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        errors: list[BaseException] = []
        for resource in (receiver, pipeline):
            try:
                resource.close()
            except BaseException as exc:
                errors.append(exc)
        if primary is None and errors:
            raise OneFrameExecutionError(
                "one-frame cleanup failed: "
                + "; ".join(f"{type(exc).__name__}: {exc}"
                             for exc in errors))

    snapshot = operational.verify_all(require_all_resolved=True)
    trace_rows = traces.verify_all()
    _require(sent is not None and outcome is not None,
             "one-frame execution did not produce evidence")
    _require(len(snapshot.outcomes) == len(trace_rows) == 1,
             "one-frame evidence count differs from one")
    _require(snapshot.outcomes[0] == outcome,
             "sealed one-frame outcome differs from the ledger")
    report = {
        "schema": SCHEMA, "run_id": request.run_id,
        "variant": request.variant.value, "transmitted_frames": 1,
        "policy_decisions": 1,
        "operational_successes": int(outcome.success),
        "operational_timeouts": int(not outcome.success),
        "ground_truth_records": 0, "live_qperc_computed": False,
        "live_reward_computed": False, "gt_used_for_ack_or_state": False,
        "clock_domain": A.CLOCK_DOMAIN,
        "config_binding_sha256": request.config_binding_sha256,
        "actor_boundary_sha256": request.actor_boundary_sha256,
        "identity_sha256": outcome.identity.exact_sha256(),
    }
    report_bytes = _canonical(report) + b"\n"
    with (request.output_root / REPORT_NAME).open("xb") as handle:
        handle.write(report_bytes)
    result = {
        "schema": RESULT_SCHEMA, "run_id": request.run_id,
        "variant": request.variant.value, "transmitted_frames": 1,
        "terminal_status": "COMPLETE",
        "result_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "config_binding_sha256": request.config_binding_sha256,
        "actor_boundary_sha256": request.actor_boundary_sha256,
    }
    with (request.output_root / RESULT_NAME).open("xb") as handle:
        handle.write(_canonical(result) + b"\n")
    return result


__all__ = [
    "SCHEMA", "RESULT_SCHEMA", "OneFrameExecutionError", "execute_one",
]

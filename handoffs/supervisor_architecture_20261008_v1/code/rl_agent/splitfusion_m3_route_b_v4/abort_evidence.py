"""Once-only Phase-6 snapshot export when exact-GT shutdown has failed.

The retained clean close remains authoritative and unchanged.  This seam only
runs after that close raises.  It snapshots already-observed state without
polling a controller or touching the GT socket, stops local telemetry readers,
and then lets the original exception propagate.  The handoff close recognizes
that recorded exception, performs one bounded abort instead of sending CLOSE,
and re-raises the same exception (with its original traceback).
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import sys
import threading

SCHEMA = "splitfusion.m3_route_b_v4.abort_evidence.v1"
ARTIFACT = "PHASE6_UE_EVIDENCE.json"


def _error(exc):
    return f"{type(exc).__name__}: {exc}"


def _atomic_write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    raw = json.dumps(document, sort_keys=True, default=str).encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def observed_snapshot(runtime, original_error):
    """Serialize observed state only; deliberately no poll, timeout or socket operation."""
    engine = runtime.engine
    controllers = list(engine.controllers)
    gt_log = getattr(runtime, "gt_log", None)
    gt_queue = getattr(runtime, "gt_queue", None)
    provider = getattr(runtime, "provider", None)
    readers = list(getattr(runtime, "readers", []))
    current = [c.current for c in controllers]
    summary = {
        "schema": "scenesense.run4_live_v2.phase6_ue_evidence.v1",
        "artifact_status": "PARTIAL_ABORTED",
        "partial": True,
        "aborted": True,
        "abort_export_schema": SCHEMA,
        "abort_error": _error(original_error),
        "export_semantics": {
            "controller_poll_performed": False,
            "handoff_close_performed_for_export": False,
            "unresolved_preserved": True,
            "missing_outcomes_inferred": False,
        },
        "coverage": engine.coverage(),
        "counters": dataclasses.asdict(engine.counters),
        "faulted": engine.faulted,
        "sessions": [c.session_uuid for c in controllers],
        "resolutions": [r.to_canonical_dict() for c in controllers for r in c.resolutions()],
        "feedback_ledgers": [c.ledger.feedback for c in controllers],
        "frames": [{**{name: getattr(f, name) for name in f.__dataclass_fields__ if name != "action"},
                    "action": f.action.to_canonical_dict(), "session_uuid": c.session_uuid}
                   for c in controllers for f in c.ledger.frames],
        "transmitted_identities": [{"frame_id": frame_id, "run4_identity": identity}
                                   for frame_id, identity in sorted(runtime._run4_identity.items())],
        "opportunities": list(engine.opportunities),
        "fallback_log": list(engine.fallback_log),
        "decisions": list(runtime.pipeline.decisions),
        "feedback_rows": list(runtime._feedback_rows),
        "terminal_rows": list(runtime._terminal_rows),
        "cycle_budget": {"frame_budget": runtime.cycle_budget.frame_budget,
                         "decision_cap": runtime.cycle_budget.decision_cap},
        "cycle_boundary_reached": bool(runtime.cycle_boundary_reached),
        "prewarm_ue_completed": bool(runtime.prewarm_report.get("completed")),
        "gt_refresh_startup": getattr(runtime, "gt_refresh_startup", None),
        "gt_objects": None if gt_log is None else gt_log.snapshot(),
        "carla_trace": getattr(runtime, "carla_trace", None),
        **runtime._run4_object_gt_evidence(),
        "gt_missing_high_outputs": None if gt_log is None else gt_log.missing_high_outputs(),
        "gt_queue": ({"counters": dict(gt_queue.counters), "depth_at_close": gt_queue.depth(),
                      "unfinished_tasks": gt_queue.unfinished_tasks} if gt_queue is not None else None),
        "unresolved_tickets_at_close": sum(1 for ticket in current
                                           if ticket is not None and ticket.resolution is None),
        "telemetry_counters": {} if provider is None else dict(provider.counters),
        "bridge_counters": {} if provider is None else dict(provider.bridge.counters),
        "reader_counters": {r.event: dict(r.counters) for r in readers},
    }
    return summary


def _stop_local(runtime, state):
    if getattr(runtime, "_m3_abort_local_stopped", False):
        return
    runtime._m3_abort_local_stopped = True
    for reader in getattr(runtime, "readers", []):
        try:
            reader.stop()
        except BaseException as exc:
            state["secondary_errors"].append("reader stop: " + _error(exc))
    audit = getattr(runtime, "audit", None)
    if audit is not None:
        try:
            audit.stop()
        except BaseException as exc:
            state["secondary_errors"].append("audit stop: " + _error(exc))


def _lock(owner):
    value = getattr(owner, "_m3_abort_lock", None)
    if value is None:
        value = threading.Lock()
        owner._m3_abort_lock = value
    return value


@contextmanager
def install_abort_evidence(*, snapshot_writer=None):
    """Install outside ``install_early_gt_ue`` so its runtime wraps this repaired base."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_exact_gt_live_v2.ue import EarlyGtUeHandoff as H
    writer = _atomic_write if snapshot_writer is None else snapshot_writer
    old_factory, old_close, old_abort = U.build_run4_runtime_class, H.close, H.abort
    if getattr(U, "_M3_ABORT_EXPORT_INSTALLED", False):
        raise RuntimeError("M3 abort exporter already installed")
    state = {"schema": SCHEMA, "snapshot_attempts": 0, "snapshot_writes": 0,
             "snapshot_write_errors": [], "handoff_aborts": 0, "secondary_errors": []}

    def factory(bindings):
        inherited = old_factory(bindings)

        class AbortEvidenceRuntime(inherited):
            def close(self):
                previous = getattr(self, "_m3_abort_failure", None)
                if previous is not None:
                    error, traceback = previous
                    raise error.with_traceback(traceback)
                try:
                    return super().close()
                except BaseException as original:
                    traceback = sys.exc_info()[2]
                    with _lock(self):
                        if getattr(self, "_m3_abort_failure", None) is None:
                            self._m3_abort_failure = (original, traceback)
                            state["snapshot_attempts"] += 1
                            _stop_local(self, state)
                            try:
                                document = observed_snapshot(self, original)
                                writer(Path(bindings.evidence_dir) / ARTIFACT, document)
                                state["snapshot_writes"] += 1
                            except BaseException as write_error:
                                message = _error(write_error)
                                state["snapshot_write_errors"].append(message)
                                self._m3_abort_snapshot_write_error = message
                            handoff = getattr(self, "_numeric_handoff", None)
                            if handoff is not None:
                                handoff._m3_abort_failure = (original, traceback)
                                handoff._m3_abort_snapshot_write_error = getattr(
                                    self, "_m3_abort_snapshot_write_error", None)
                    raise original.with_traceback(traceback)

        AbortEvidenceRuntime.__name__ = "M3V4AbortEvidence" + inherited.__name__
        return AbortEvidenceRuntime

    def abort(handoff, exc):
        with _lock(handoff):
            if getattr(handoff, "_m3_abort_done", False):
                return None
            handoff._m3_abort_done = True
            state["handoff_aborts"] += 1
        return old_abort(handoff, exc)

    def close(handoff):
        failure = getattr(handoff, "_m3_abort_failure", None)
        if failure is None:
            return old_close(handoff)
        error, traceback = failure
        abort(handoff, error)
        raise error.with_traceback(traceback)

    U.build_run4_runtime_class, H.close, H.abort = factory, close, abort
    U._M3_ABORT_EXPORT_INSTALLED = True
    try:
        yield state
    finally:
        U.build_run4_runtime_class, H.close, H.abort = old_factory, old_close, old_abort
        del U._M3_ABORT_EXPORT_INSTALLED

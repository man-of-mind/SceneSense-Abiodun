"""UE-child retention wiring for the retention launcher (uses ``splitfusion_paired_retention_v2`` unchanged).

``AttemptRetentionObserver`` adds two things to ``PairedRetentionObserverV2`` and nothing else:
  * the ``threading.Thread`` object that performed the first policy-decision materialization (the owned UE frame
    worker, ``route-b-split-front``). Every later materialization on another thread is already a counted
    FOREIGN_THREAD fault in the base class. ``finalize`` is called with exactly this thread as the drain proof;
  * bounded per-call wall durations (``perf_counter_ns``) of its own ``begin``/``observe_input``/``end`` bookkeeping,
    so the decision-path overhead of retention is measured, not assumed. No I/O, copy, hash or sync is added.

``finalize_after_child`` runs only after ``CHILD.main`` has returned and every production scope has exited. It never
raises and never changes the child's return code: retention problems are written to ``RETENTION_RESULT.json`` (and are
reported as missing samples), not to stderr, so they cannot be confused with a production failure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import traceback

from rl_agent.splitfusion_paired_retention_v2 import observer as OB
from rl_agent.splitfusion_paired_retention_v2 import plan as RP

from . import contract as C

MAX_TIMINGS = 4096
SCHEMA = "splitfusion.m3_route_b_v2.retention_result.v1"


class AttemptRetentionObserver(OB.PairedRetentionObserverV2):
    def __init__(self, config, directory=None):
        super().__init__(config, directory)
        self.owner_thread = None
        self.timings_ns = {"begin": [], "observe_input": [], "end": []}
        self.timings_dropped = 0

    def _time(self, name, start):
        series = self.timings_ns[name]
        if len(series) < MAX_TIMINGS:
            series.append(time.perf_counter_ns() - start)
        else:
            self.timings_dropped += 1

    def begin(self, planned):
        start = time.perf_counter_ns()
        token = super().begin(planned)
        if self.owner_thread is None and self._owner == threading.get_ident():
            self.owner_thread = threading.current_thread()
        self._time("begin", start)
        return token

    def observe_input(self, tensor, device):
        start = time.perf_counter_ns()
        try:
            return super().observe_input(tensor, device)
        finally:
            self._time("observe_input", start)

    def end(self, token, prepared=None):
        start = time.perf_counter_ns()
        try:
            return super().end(token, prepared)
        finally:
            self._time("end", start)


def load_observer(output: Path, registration: dict) -> AttemptRetentionObserver:
    """Bind the sealed registration's pre-launch plan file (digest-checked) to an enabled observer."""
    section = registration.get("paired_retention") or {}
    if section != C.retention_registration(section.get("plan_sha256")):
        raise ValueError("registration retention section differs from the contract")
    config = RP.RetentionConfigV2.from_plan_file(Path(output) / C.RETENTION_PLAN_FILE, section["plan_sha256"])
    if RP.canonical(config.plan) != RP.canonical(C.retention_plan(registration["transmitted_budget"])):
        raise ValueError("retention plan is not the registered rule for this budget")
    return AttemptRetentionObserver(config)


def _summary(series):
    if not series:
        return dict(n=0)
    ordered = sorted(series)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
    return dict(n=len(ordered), p50_us=pick(.5) / 1e3, p95_us=pick(.95) / 1e3, p99_us=pick(.99) / 1e3,
                max_us=ordered[-1] / 1e3, sum_ms=sum(ordered) / 1e6)


def finalize_after_child(observer, output: Path, *, child_returncode, child_error) -> dict:
    """Post-drain persistence + result record. Never raises."""
    output = Path(output)
    result = dict(schema=SCHEMA, child_returncode=child_returncode, child_error=child_error,
                  owner_thread=None, finalized=False, error=None, manifest_sha256=None)
    started = time.monotonic()
    try:
        thread = observer.owner_thread
        result["owner_thread"] = None if thread is None else dict(name=thread.name, ident=thread.ident,
                                                                  alive=thread.is_alive(), daemon=thread.daemon)
        workers = [] if thread is None else [thread]
        manifest = observer.finalize(output / C.RETENTION_DIR, workers=workers)
        result.update(finalized=True, manifest_sha256=manifest["manifest_sha256"],
                      status_counts=manifest["status_counts"], counters=manifest["counters"],
                      observed_policy_decisions=manifest["observed_policy_decisions"],
                      run_length=manifest["run_length"], host_bytes=manifest["host_bytes"],
                      storage_bytes=manifest["storage_bytes"], faults=manifest["faults"])
    except BaseException as exc:  # noqa: BLE001 - never alter the child's outcome
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        result["traceback"] = traceback.format_exc()[-4000:]
        result["counters"] = dict(getattr(observer, "counters", {}))
    result["finalize_wall_s"] = time.monotonic() - started
    result["decision_path_bookkeeping"] = {k: _summary(v) for k, v in observer.timings_ns.items()}
    result["decision_path_timings_dropped"] = observer.timings_dropped
    result["peak_host_bytes"] = getattr(observer, "peak_host_bytes", None)
    try:
        raw = json.dumps(result, indent=1, sort_keys=True, default=str).encode()
        with open(output / C.RETENTION_RESULT_FILE, "xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:  # noqa: BLE001
        pass
    return result

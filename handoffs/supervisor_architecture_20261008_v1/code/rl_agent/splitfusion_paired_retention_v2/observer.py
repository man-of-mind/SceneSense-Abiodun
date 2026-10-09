"""Opt-in bounded paired-input retention observer (v2). Disabled by default.

Decision-path contract (``begin``/``observe_input``/``end``/``notify_feedback``): reference capture
and integer bookkeeping only. No clone, no D2H copy, no tensor scan, no hashing, no serialization, no
file or socket I/O and no device synchronization. These methods NEVER raise into production code:
every problem is recorded as a fault and as a per-ordinal status, and the run continues.

Sampling: a policy decision is retained only if its route-level ordinal is in the pre-launch plan
(``plan.py``). Off-sample decisions are counted and skipped; a full budget is a counted
``BUDGET_EXHAUSTED`` status, never an exception; a shorter run leaves ``NOT_REACHED`` ordinals and a
longer run counts ``SKIPPED_BEYOND_PLAN`` decisions without resampling.

Persistence happens either after drain (``POST_DRAIN``; every sampled record stays referenced, host
bytes bounded by the plan cap) or in one background writer thread (``BACKGROUND_SPILL``) that receives
a record only after its feedback was notified or after ``spill_lag_decisions`` later decisions began,
so RAM stays bounded to a small window. Retaining host memory changes allocation pressure: a run with
this enabled is a quality diagnostic, not an uninstrumented timing measurement.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Optional

import torch

from .plan import INPUT_BYTES, INPUT_SHAPE, MAX_WIRE_BYTES, RetentionConfigV2, canonical, plan_sha256
from . import writer as WR

SCHEMA = "splitfusion.paired_retention.archive.v2"
MAX_FAULT_DETAILS = 64
MAX_IDENTITY_LEDGER = 16384
FINAL_PREFIXES = ("RETAINED_COMPLETE", "INCOMPLETE_", "BUDGET_EXHAUSTED", "FAULT_", "NOT_REACHED")
PACKAGE_SOURCES = ("plan.py", "observer.py", "writer.py")

# Seams this observer would wrap (verified at install, never edited).
SEAM_PINS = {
    "rl_agent/splitfusion_ue_input_prep_v2/candidate.py":
        "565ce379045836045d9111cfc8eb93a9c22c9bcf3544380ae7854d623753439f",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_ue_runtime_v2.py":
        "c97103645d4e55dea3a35a7be8b942ec15d1320223d83996c04bcb5d5ebc3e66",
    "rl_agent/splitfusion_live_dispatch_v1/live_pilot_runtime.py":
        "ce568a89fa262916e7b6f737a5d1f7fc310428de96ff170eb56eca303411f894",
}
FUSED_RETURN = "return out.to(device=device, dtype=torch.float32)"
FUSED_CAPTURE = ("return _paired_retention_v2_observer(out, device)"
                 ".to(device=device, dtype=torch.float32)")
ROOT = Path(__file__).resolve().parents[2]


def package_source_digests() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: WR.sha256_file(here / name) for name in PACKAGE_SOURCES}


class NoOpRetentionObserver:
    """What a disabled config yields: identity pass-through, no state, no threads."""
    enabled = False

    def begin(self, planned: Any) -> None:
        return None

    def observe_input(self, tensor: Any, device: Any) -> Any:
        return tensor

    def end(self, token: Any, prepared: Any = None) -> None:
        return None

    def notify_feedback(self, frame_id: int) -> None:
        return None

    def start(self) -> dict:
        return {"enabled": False}

    def finalize(self, *args: Any, **kwargs: Any) -> dict:
        return {"enabled": False, "records": 0}


class PairedRetentionObserverV2:
    enabled = True

    def __init__(self, config: RetentionConfigV2, directory: Optional[Path] = None):
        if not isinstance(config, RetentionConfigV2) or not config.enabled:
            raise ValueError("PairedRetentionObserverV2 requires an explicitly enabled config")
        self.plan, self.plan_sha256 = config.plan, config.plan_sha256
        if plan_sha256(self.plan) != self.plan_sha256:
            raise ValueError("plan digest drift")
        bounds, persistence = self.plan["bounds"], self.plan["persistence"]
        self.mode = persistence["mode"]
        self.max_records, self.max_host_bytes = bounds["max_records"], bounds["max_host_bytes"]
        self.lag = persistence["spill_lag_decisions"]
        self.require_cuda = self.plan["require_cuda_transfer"]
        self.ordinals = frozenset(self.plan["ordinals"])
        self.planned_decisions = self.plan["planned_policy_decisions"]
        self.directory = None if directory is None else Path(directory)
        self.writer: Optional[WR.BackgroundWriter] = None
        if self.mode == "BACKGROUND_SPILL":
            if self.directory is None:
                raise ValueError("BACKGROUND_SPILL requires the retention directory at construction")
            self.writer = WR.BackgroundWriter(self.directory, max_storage_bytes=bounds["max_storage_bytes"],
                min_free_disk_bytes=bounds["min_free_disk_bytes"],
                queue_records=persistence["spill_queue_records"], require_cuda=self.require_cuda)
        self.status: dict[int, str] = {}
        self.details: dict[int, str] = {}
        self.records: dict[int, dict] = {}
        self._held: deque = deque()
        self._feedback: deque = deque(maxlen=4096)
        self.counters = {"policy_decisions_observed": 0, "non_decision_frames": 0, "sampled": 0,
                         "skipped_off_sample": 0, "skipped_beyond_plan": 0, "budget_exhausted": 0,
                         "faults": 0, "faulted_unsampled": 0,
                         "spill_backpressure_deferrals": 0}
        self.fault_details: list[dict] = []
        self._ordinal = 0
        self._reserved = self._unreserved = self.peak_host_bytes = 0
        self._retained_count = 0
        self._owner: Optional[int] = None
        self._active: Optional[dict] = None
        self._local = threading.local()
        self._keys: set = set()
        self.identity_ledger_saturated = False
        self._started = self._closed = False

    # ------------------------------------------------------------------ bookkeeping (no I/O)
    def _fault(self, reason: str, ordinal: Optional[int] = None, exc: Optional[BaseException] = None) -> None:
        self.counters["faults"] += 1
        if len(self.fault_details) < MAX_FAULT_DETAILS:
            detail = {"reason": reason, "ordinal": ordinal}
            if exc is not None:
                detail["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.fault_details.append(detail)

    def _mark(self, rec: dict, status: str, detail: str = "") -> None:
        """Mark a sampled record incomplete and drop its references (never raises)."""
        if self.status.get(rec["ordinal"]) in ("HELD", "OPEN"):   # never touch writer-owned records
            self.status[rec["ordinal"]] = status
            if detail:
                self.details[rec["ordinal"]] = detail[:300]
            self._unreserved += rec.get("reserved_bytes", 0)
            rec["reserved_bytes"] = 0
            rec["input"] = rec["prepared"] = None
            for index, held in enumerate(self._held):     # identity, not dict equality
                if held is rec:
                    del self._held[index]
                    break

    def host_bytes(self) -> int:
        released = 0 if self.writer is None else self.writer.released_bytes()
        return self._reserved - self._unreserved - released

    def _spill_tick(self, ordinal: int) -> None:
        if self.writer is None:
            return
        done = set()
        while self._feedback:
            done.add(self._feedback.popleft())
        while self._held:
            rec = self._held[0]
            if not (rec["ordinal"] <= ordinal - self.lag or rec["frame_id"] in done):
                break
            if not self.writer.submit(rec):
                self.counters["spill_backpressure_deferrals"] += 1
                break
            self._held.popleft()
            self.status[rec["ordinal"]] = "SPILL_QUEUED"

    # ------------------------------------------------------------------ decision path
    def begin(self, planned: Any) -> Optional[dict]:
        try:
            if not getattr(planned, "reward_requested", False):
                self.counters["non_decision_frames"] += 1
                return None          # holds / fallbacks / warm-up: no ordinal
            kind = getattr(getattr(getattr(planned, "plan", None), "kind", None), "value", None)
            if kind != "POLICY_DECISION":
                self._fault("FOREIGN_REWARD_KIND")
                return None
            tid = threading.get_ident()
            if self._owner is None:
                self._owner = tid
            elif self._owner != tid:
                self._fault("FOREIGN_THREAD")
                return None
            if self._closed:
                self._fault("BEGIN_AFTER_FINALIZE")
                return None
            ordinal = self._ordinal
            self._ordinal += 1
            self.counters["policy_decisions_observed"] += 1
            self._spill_tick(ordinal)
            sampled = ordinal in self.ordinals
            if self._active is not None:          # reentrant materialization: both are faults
                self._fault("REENTRANT_MATERIALIZATION", ordinal)
                self._mark(self._active, "INCOMPLETE_REENTRANT")
                if sampled:
                    self.status[ordinal] = "FAULT_REENTRANT"
                else:
                    self.counters["faulted_unsampled"] += 1
                return None
            key = (planned.session_uuid, planned.tensor_seq, planned.frame_id)
            if key in self._keys:
                self._fault("DUPLICATE_IDENTITY", ordinal)
                if sampled:
                    self.status[ordinal] = "FAULT_DUPLICATE_IDENTITY"
                else:
                    self.counters["faulted_unsampled"] += 1
                return None
            if len(self._keys) < MAX_IDENTITY_LEDGER:
                self._keys.add(key)
            else:
                self.identity_ledger_saturated = True
            if not sampled:
                if ordinal >= self.planned_decisions:
                    self.counters["skipped_beyond_plan"] += 1
                else:
                    self.counters["skipped_off_sample"] += 1
                return None
            self.counters["sampled"] += 1
            if (self._retained_count >= self.max_records
                    or self.host_bytes() + INPUT_BYTES > self.max_host_bytes):
                self.counters["budget_exhausted"] += 1
                self.status[ordinal] = "BUDGET_EXHAUSTED"
                return None
            rec = {"ordinal": ordinal, "planned": planned, "frame_id": planned.frame_id,
                   "input": None, "device": None, "prepared": None, "reserved_bytes": INPUT_BYTES}
            self._reserved += INPUT_BYTES
            self.peak_host_bytes = max(self.peak_host_bytes, self.host_bytes())
            self._retained_count += 1
            self.status[ordinal] = "OPEN"
            self.records[ordinal] = rec
            self._active = rec
            self._local.current = rec
            return rec
        except Exception as exc:  # noqa: BLE001 - observer faults never reach production
            self._fault("BEGIN_EXCEPTION", exc=exc)
            return None

    def observe_input(self, tensor: Any, device: Any) -> Any:
        rec = getattr(self._local, "current", None)
        if rec is None:
            return tensor     # unsampled, hold, fallback, warm-up or another thread
        try:
            if rec["input"] is not None:
                self._fault("SECOND_INPUT_IN_ONE_MATERIALIZATION", rec["ordinal"])
                self._mark(rec, "INCOMPLETE_MULTIPLE_INPUTS")
            elif (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != INPUT_SHAPE
                    or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                    or tensor.requires_grad or not tensor.is_contiguous()):
                self._fault("BAD_INPUT_TENSOR", rec["ordinal"])
                self._mark(rec, "INCOMPLETE_BAD_INPUT_TENSOR")
            elif self.status.get(rec["ordinal"]) == "OPEN":
                rec["input"] = tensor              # by reference: fresh per-call host storage
                rec["device"] = str(device)
        except Exception as exc:  # noqa: BLE001
            self._fault("OBSERVE_EXCEPTION", rec.get("ordinal"), exc)
            self._mark(rec, "INCOMPLETE_OBSERVER_EXCEPTION")
        return tensor

    def end(self, token: Optional[dict], prepared: Any = None) -> None:
        if token is None:
            return
        try:
            if self.status.get(token["ordinal"]) != "OPEN":
                pass                                   # already classified
            elif getattr(self._local, "current", None) is not token:
                self._fault("CONTEXT_CHANGED", token["ordinal"])
                self._mark(token, "INCOMPLETE_CONTEXT_CHANGED")
            elif prepared is None:
                self._mark(token, "INCOMPLETE_PRODUCTION_EXCEPTION")
            elif token["input"] is None:
                self._mark(token, "INCOMPLETE_NO_INPUT_OBSERVED")
            else:
                wire = getattr(prepared, "wire", None)
                if type(wire) is not bytes or not wire:
                    self._mark(token, "INCOMPLETE_NO_WIRE")
                elif (len(wire) > MAX_WIRE_BYTES
                        or self.host_bytes() + len(wire) > self.max_host_bytes):
                    self.counters["budget_exhausted"] += 1
                    self._mark(token, "INCOMPLETE_WIRE_BUDGET")
                else:
                    token["prepared"] = {"wire": wire, "identity": dict(prepared.identity),
                                         "gt_identity": dict(prepared.gt_identity)}
                    token["reserved_bytes"] += len(wire)
                    self._reserved += len(wire)
                    self.peak_host_bytes = max(self.peak_host_bytes, self.host_bytes())
                    self.status[token["ordinal"]] = "HELD"
                    self._held.append(token)
        except Exception as exc:  # noqa: BLE001
            self._fault("END_EXCEPTION", token.get("ordinal"), exc)
            self._mark(token, "INCOMPLETE_OBSERVER_EXCEPTION")
        finally:
            self._local.current = None
            if self._active is token:
                self._active = None

    def notify_feedback(self, frame_id: int) -> None:
        """Optional: the decision's feedback/terminal arrived; its record may now spill."""
        try:
            self._feedback.append(int(frame_id))
        except Exception as exc:  # noqa: BLE001
            self._fault("FEEDBACK_NOTIFY_EXCEPTION", exc=exc)

    # ------------------------------------------------------------------ lifecycle (off path)
    def start(self) -> dict:
        if self._started:
            raise ValueError("observer already started")
        self._started = True
        if self.writer is not None:
            self.writer.start()
        return {"enabled": True, "plan_sha256": self.plan_sha256, "mode": self.mode,
                "max_records": self.max_records, "max_host_bytes": self.max_host_bytes,
                "writer_thread": self.writer is not None}

    def finalize(self, directory: Optional[Path] = None, *, workers: Iterable[Any]) -> dict:
        """After every owned producer thread has drained: persist, classify, seal the manifest."""
        drain = verified_drain(workers)
        if self._closed:
            raise ValueError("observer already finalized")
        if self._active is not None:
            raise ValueError("finalize while a materialization is open")
        self._closed = True
        if self.mode == "BACKGROUND_SPILL":
            directory = self.directory
            while self._held:
                rec = self._held.popleft()
                self.status[rec["ordinal"]] = "SPILL_QUEUED"
                self.writer.queue.put(rec)        # post-drain: blocking put is fine
            results = self.writer.close()
        else:
            if directory is None:
                raise ValueError("POST_DRAIN finalize requires a fresh directory")
            directory = Path(directory)
            if any(p.is_symlink() for p in (directory, *directory.parents)):
                raise ValueError("retention path must not use symlinks")
            directory.mkdir(parents=False, exist_ok=False)
            budget = WR.StorageBudget(directory, self.plan["bounds"]["max_storage_bytes"],
                                      self.plan["bounds"]["min_free_disk_bytes"])
            results = {}
            while self._held:
                rec = self._held.popleft()
                try:
                    results[rec["ordinal"]] = WR.persist_record(rec, directory, budget,
                                                                require_cuda=self.require_cuda)
                except WR.RetentionWriteError as exc:
                    results[rec["ordinal"]] = {"ordinal": rec["ordinal"], "status": exc.reason,
                                               "detail": str(exc)[:300]}
                self._unreserved += rec["reserved_bytes"]
                WR.release(rec)
        observed = self._ordinal
        entries, incomplete = [], []
        for ordinal in self.plan["ordinals"]:
            status = self.status.get(ordinal)
            if ordinal in results:
                status = results[ordinal]["status"]
                if status == "RETAINED_COMPLETE":
                    entries.append(results[ordinal])
                else:
                    self.details[ordinal] = results[ordinal].get("detail", "")
            elif status is None:
                status = "NOT_REACHED" if ordinal >= observed else "FAULT_UNOBSERVED_SAMPLED_ORDINAL"
            elif status in ("OPEN", "HELD", "SPILL_QUEUED"):
                status = "INCOMPLETE_NOT_PERSISTED"
            self.status[ordinal] = status
            if status != "RETAINED_COMPLETE":
                incomplete.append({"ordinal": ordinal, "status": status,
                                   "detail": self.details.get(ordinal, "")})
        for rec in self.records.values():
            WR.release(rec)
        self.records.clear()
        counts: dict[str, int] = {}
        for ordinal in self.plan["ordinals"]:
            counts[self.status[ordinal]] = counts.get(self.status[ordinal], 0) + 1
        files = {}
        for entry in entries:
            files[entry["input_file"]] = entry["input_sha256"]
            files[entry["wire_file"]] = entry["wire_sha256"]
        planned = self.planned_decisions
        manifest = {
            "schema": SCHEMA, "complete": True, "plan": self.plan, "plan_sha256": self.plan_sha256,
            "observed_policy_decisions": observed,
            "run_length": ("SHORTER_THAN_PLANNED" if observed < planned else
                           "LONGER_THAN_PLANNED" if observed > planned else "EQUAL_TO_PLANNED"),
            "ordinal_status": {str(o): self.status[o] for o in self.plan["ordinals"]},
            "status_counts": counts, "counters": dict(self.counters),
            "records": entries, "not_retained": incomplete,
            "faults": {"count": self.counters["faults"], "details": list(self.fault_details)},
            "identity_ledger_saturated": self.identity_ledger_saturated,
            "host_bytes": {"peak": self.peak_host_bytes, "cap": self.max_host_bytes,
                           "at_close": self.host_bytes()},
            "storage_bytes": sum(e["input_file_bytes"] + e["wire_bytes"] for e in entries),
            "sha256": files, "drain": drain, "sources": package_source_digests(),
            "persistence_mode": self.mode,
            "claim_scope": self.plan["claim_scope"],
            "decision_path": {"disk_writes": 0, "clones": 0, "d2h_copies": 0, "device_syncs": 0,
                              "hashing": 0},
            "gt_policy": "REFERENCES_TO_THE_NUMERIC_GT_ARCHIVE_ONLY__RESOLVED_POSTFLIGHT",
        }
        raw = canonical(manifest)
        path = directory / "MANIFEST.json"
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return dict(manifest, manifest_sha256=hashlib.sha256(raw).hexdigest())


def verified_drain(workers: Iterable[Any]) -> dict:
    workers = list(workers)
    if not workers:
        raise ValueError("at least one owned producer worker must be shown drained")
    evidence = {}
    for index, worker in enumerate(workers):
        alive = getattr(worker, "is_alive", None)
        if not callable(alive) or alive() is not False:
            raise ValueError(f"worker {index} has not drained")
        evidence[str(index)] = {"alive": False, "name": str(getattr(worker, "name", ""))[:80]}
    return evidence


def make_observer(config: RetentionConfigV2 = RetentionConfigV2(), directory: Optional[Path] = None):
    """Disabled (default) -> no-op; enabled -> bounded observer bound to the pre-launch plan."""
    if not isinstance(config, RetentionConfigV2):
        raise TypeError("RetentionConfigV2 required")
    return PairedRetentionObserverV2(config, directory) if config.enabled else NoOpRetentionObserver()


# ---------------------------------------------------------------------- seam (not installed anywhere)
def retained_fused_packing(original, observe):
    """Insert an identity observer before the single final CPU->device copy of the v2 fused input prep."""
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        raise ValueError("input preparation source unavailable (already rewritten?)") from exc
    if source.count(FUSED_RETURN) != 1 or "_paired_retention_v2_observer" in source:
        raise ValueError("unsupported or already instrumented input preparation")
    namespace = dict(original.__globals__, _paired_retention_v2_observer=observe)
    exec(compile(source.replace(FUSED_RETURN, FUSED_CAPTURE), "<paired-retention-v2-input>", "exec"), namespace)
    return namespace[original.__name__]


_SCOPE_LOCK = threading.Lock()


@contextmanager
def install_retention(observer):
    """Enter INSIDE install_fused_candidate() and install_ue_pose() (innermost); restores on exit.

    Disabled observers install nothing. No launcher calls this; a new prospective registration must
    bind it (see INTEGRATION.md) and call observer.finalize(...) only after the producer drains.
    """
    if not getattr(observer, "enabled", False):
        yield {"enabled": False}
        return
    if not _SCOPE_LOCK.acquire(blocking=False):
        raise ValueError("nested retention scope")
    try:
        for relative, digest in SEAM_PINS.items():
            if WR.sha256_file(ROOT / relative) != digest:
                raise ValueError("retention seam source drift: " + relative)
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
        from rl_agent.splitfusion_ue_input_prep_v2 import candidate as V2
        if B._prepare_live_input is not V2.prepare_live_input_fused:
            raise ValueError("v2 fused input candidate is not installed; enter install_fused_candidate() first")
        packing, materialize = B._prepare_live_input, U.Run4FramePipelineV2.materialize
        replacement = retained_fused_packing(packing, observer.observe_input)

        def observed(self, planned, *args, **kwargs):
            token = prepared = None
            try:
                token = observer.begin(planned)    # never raises
                prepared = materialize(self, planned, *args, **kwargs)
                return prepared
            finally:
                observer.end(token, prepared)      # never raises; production exception preserved

        started = observer.start()
        B._prepare_live_input = replacement
        U.Run4FramePipelineV2.materialize = observed
        try:
            yield dict(started, seam_pins=dict(SEAM_PINS))
        finally:
            if B._prepare_live_input is replacement:
                B._prepare_live_input = packing
            if U.Run4FramePipelineV2.materialize is observed:
                U.Run4FramePipelineV2.materialize = materialize
    finally:
        _SCOPE_LOCK.release()

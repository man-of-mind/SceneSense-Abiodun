"""Reversible Linux thread affinities for an explicitly enumerated set of PIDs.

No scheduler policy/priority setters are used. Each identity is checked using
/proc starttime immediately before and after writes. Linux does not offer an
atomic identity-check-and-affinity-set operation; a reuse race inside that
interval cannot be eliminated by this interface. There is no background watcher:
convergence is a bounded observation and later thread births need a fresh pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional


def cpu_set(values: Iterable[int]) -> tuple[int, ...]:
    """Normalize an explicit, nonempty mask without silently dropping CPUs."""
    raw = tuple(values)
    if not raw or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                      for cpu in raw):
        raise ValueError("CPU masks must contain nonnegative integer CPU IDs")
    return tuple(sorted(set(raw)))


def validate_disjoint_cpu_sets(partitions: Mapping[str, Iterable[int]],
                               available_cpus: Iterable[int]) -> dict[str, tuple[int, ...]]:
    """Validate topology supplied by the caller; never choose a partition."""
    available = set(cpu_set(available_cpus))
    result = {name: cpu_set(values) for name, values in partitions.items()}
    used: set[int] = set()
    for name, values in result.items():
        if not set(values) <= available:
            raise ValueError(f"{name}: unavailable CPUs {sorted(set(values) - available)}")
        if used.intersection(values):
            raise ValueError(f"{name}: overlapping CPUs {sorted(used.intersection(values))}")
        used.update(values)
    return result


@dataclass(frozen=True)
class ProcessPlan:
    pid: int
    cpus: tuple[int, ...]
    newborn_restore_cpus: tuple[int, ...]
    expected_starttime: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("An explicit positive PID is required")
        object.__setattr__(self, "cpus", cpu_set(self.cpus))
        object.__setattr__(self, "newborn_restore_cpus", cpu_set(self.newborn_restore_cpus))


@dataclass(frozen=True)
class TaskSnapshot:
    pid: int
    pid_starttime: int
    tid: int
    tid_starttime: int
    comm: str
    affinity: tuple[int, ...]
    policy: int
    priority: int


class IdentityChanged(RuntimeError):
    pass


class PartitionError(RuntimeError):
    """Apply failed; ``report`` includes the outcome of automatic rollback."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


class ProcBackend:
    """Inject a proc root and os-like API, or replace this backend in tests."""

    def __init__(self, proc_root: str = "/proc", os_api=os):
        self.proc_root = Path(proc_root)
        self.os_api = os_api

    @staticmethod
    def parse_stat(value: str) -> tuple[int, str]:
        # comm may contain spaces and parentheses. Fields after its LAST closing
        # parenthesis begin at stat field 3; starttime is stat field 22.
        left, right = value.index("("), value.rindex(")")
        return int(value[right + 2:].split()[19]), value[left + 1:right]

    def process_starttime(self, pid: int) -> int:
        return self.parse_stat((self.proc_root / str(pid) / "stat").read_text())[0]

    def list_tids(self, pid: int) -> list[int]:
        return sorted(int(item.name) for item in
                      (self.proc_root / str(pid) / "task").iterdir() if item.name.isdigit())

    def task_identity(self, pid: int, tid: int) -> tuple[int, str]:
        return self.parse_stat((self.proc_root / str(pid) / "task" / str(tid) / "stat").read_text())

    def snapshot(self, pid: int, tid: int) -> TaskSnapshot:
        proc_start = self.process_starttime(pid)
        task_start, comm = self.task_identity(pid, tid)
        affinity = tuple(sorted(self.os_api.sched_getaffinity(tid)))
        policy = self.os_api.sched_getscheduler(tid)
        priority = self.os_api.sched_getparam(tid).sched_priority
        if (proc_start != self.process_starttime(pid)
                or task_start != self.task_identity(pid, tid)[0]):
            raise IdentityChanged(f"PID {pid}/TID {tid} changed while reading")
        return TaskSnapshot(pid, proc_start, tid, task_start, comm, affinity, policy, priority)

    def set_affinity(self, tid: int, cpus: Iterable[int]) -> None:
        self.os_api.sched_setaffinity(tid, set(cpus))


def _gone(error: BaseException) -> bool:
    return isinstance(error, ProcessLookupError) or (
        isinstance(error, OSError) and error.errno in (errno.ENOENT, errno.ESRCH))


def _key(snapshot: TaskSnapshot) -> str:
    return f"{snapshot.tid}:{snapshot.tid_starttime}"


class PartitionSession:
    """Apply, serialize, and restore a finite set of thread affinity changes.

    ``persist`` receives a complete JSON-serializable journal before every mask
    write and on completion. The caller should persist it atomically if crash
    recovery is needed. ``from_dict`` reconstructs a session for later restore.
    Newly observed threads use the required per-process restoration mask because
    their observed mask may already have been inherited from a modified parent.
    """

    def __init__(self, backend=None, *, max_passes: int = 5,
                 persist: Optional[Callable[[dict], None]] = None):
        if max_passes < 1:
            raise ValueError("max_passes must be positive")
        self.backend = backend if backend is not None else ProcBackend()
        self.max_passes = max_passes
        self.persist = persist
        self.processes: list[dict] = []
        self.events: list[dict] = []
        self.status = "new"

    def to_dict(self) -> dict:
        # Round-trip copying prevents persistence callbacks mutating live state.
        import json
        return json.loads(json.dumps({"version": 1, "max_passes": self.max_passes,
                                      "status": self.status, "processes": self.processes,
                                      "events": self.events}))

    @classmethod
    def from_dict(cls, value: dict, backend=None, *, persist=None):
        import json
        value = json.loads(json.dumps(value))
        if value.get("version") != 1:
            raise ValueError("Unsupported partition journal version")
        session = cls(backend, max_passes=value["max_passes"], persist=persist)
        session.status = value["status"]
        session.processes = value["processes"]
        session.events = value["events"]
        # Validate all masks/targets before a reconstructed session can write.
        pids = set()
        for process in session.processes:
            plan = ProcessPlan(process["pid"], process["target_cpus"],
                               process["newborn_restore_cpus"], process["pid_starttime"])
            if plan.pid in pids:
                raise ValueError("Duplicate PID in journal")
            pids.add(plan.pid)
            for key, entry in process["threads"].items():
                before = entry["before"]
                if (before["pid"] != plan.pid or before["pid_starttime"] != plan.expected_starttime
                        or key != f'{before["tid"]}:{before["tid_starttime"]}'):
                    raise ValueError("Thread identity does not match journal process")
                cpu_set(entry["restore_cpus"])
        return session

    def _persist(self):
        if self.persist is not None:
            self.persist(self.to_dict())

    def _scan(self, process: dict) -> list[TaskSnapshot]:
        pid = process["pid"]
        if self.backend.process_starttime(pid) != process["pid_starttime"]:
            raise IdentityChanged(f"PID {pid} was reused")
        result = []
        for tid in self.backend.list_tids(pid):
            try:
                snap = self.backend.snapshot(pid, tid)
                if snap.pid_starttime != process["pid_starttime"]:
                    raise IdentityChanged(f"PID {pid} was reused")
                result.append(snap)
            except OSError as error:
                if not _gone(error):
                    raise
        if self.backend.process_starttime(pid) != process["pid_starttime"]:
            raise IdentityChanged(f"PID {pid} was reused")
        return result

    @staticmethod
    def _remember(process: dict, snapshot: TaskSnapshot, baseline=False) -> dict:
        key = _key(snapshot)
        if key not in process["threads"]:
            process["threads"][key] = {
                "before": asdict(snapshot), "baseline": baseline, "attempted": False,
                "restore_cpus": list(snapshot.affinity if baseline
                                     else process["newborn_restore_cpus"]),
            }
        return process["threads"][key]

    def _write(self, process: dict, snapshot: TaskSnapshot, cpus, mode: str):
        entry = self._remember(process, snapshot)
        check = self.backend.snapshot(snapshot.pid, snapshot.tid)
        if (check.pid_starttime, check.tid_starttime) != (snapshot.pid_starttime, snapshot.tid_starttime):
            raise IdentityChanged(f"PID {snapshot.pid}/TID {snapshot.tid} changed before write")
        if check.affinity == tuple(cpus):
            return
        entry["attempted"] = True
        event = {"mode": mode, "pid": snapshot.pid, "tid": snapshot.tid,
                 "pid_starttime": snapshot.pid_starttime, "tid_starttime": snapshot.tid_starttime,
                 "before": asdict(check), "requested_cpus": list(cpus), "outcome": "pending"}
        self.events.append(event)
        self._persist()
        try:
            # Journal I/O may have taken time: recheck after it, directly before
            # the syscall. Never intentionally target a replacement identity.
            check = self.backend.snapshot(snapshot.pid, snapshot.tid)
            if (check.pid_starttime, check.tid_starttime) != (snapshot.pid_starttime, snapshot.tid_starttime):
                raise IdentityChanged(f"PID {snapshot.pid}/TID {snapshot.tid} changed before write")
            self.backend.set_affinity(snapshot.tid, cpus)
            after = self.backend.snapshot(snapshot.pid, snapshot.tid)
            event["after"] = asdict(after)
            if (after.pid_starttime, after.tid_starttime) != (snapshot.pid_starttime, snapshot.tid_starttime):
                raise IdentityChanged(f"PID {snapshot.pid}/TID {snapshot.tid} changed during write")
            if after.affinity != tuple(cpus):
                raise RuntimeError(f"TID {snapshot.tid}: requested affinity was not applied exactly")
            if (after.policy, after.priority) != (check.policy, check.priority):
                raise RuntimeError(f"TID {snapshot.tid}: scheduler settings changed during affinity write")
            event["outcome"] = "verified"
        except BaseException as error:
            event["outcome"] = "failed"
            event["error"] = f"{type(error).__name__}: {error}"
            raise

    def _converge(self, mode: str, *, tolerate_process_exit=False) -> dict:
        report = {"mode": mode, "before": [], "after": [], "skipped": [], "errors": [],
                  "passes": 0, "converged": False}
        excluded = set()
        skipped_threads = set()

        def reused_thread(process, snap):
            identity = (snap.pid, snap.tid, snap.tid_starttime)
            if identity in skipped_threads:
                return True
            if (mode != "apply" and _key(snap) not in process["threads"]
                    and any(entry["before"]["tid"] == snap.tid
                            for entry in process["threads"].values())):
                skipped_threads.add(identity)
                report["skipped"].append({"pid": snap.pid, "tid": snap.tid,
                                           "reason": "TID identity changed; affinity not touched"})
                return True
            return False

        for pass_index in range(self.max_passes):
            stable = True
            after_all = []
            for process in self.processes:
                if process["pid"] in excluded:
                    continue
                try:
                    before = self._scan(process)
                    if pass_index == 0:
                        report["before"].extend(asdict(snap) for snap in before)
                    for snap in before:
                        if reused_thread(process, snap):
                            continue
                        entry = self._remember(process, snap)
                        cpus = (process["target_cpus"] if mode == "apply" else entry["restore_cpus"])
                        try:
                            self._write(process, snap, cpus, mode)
                        except OSError as error:
                            if _gone(error):
                                stable = False
                            elif mode == "apply":
                                raise
                            else:
                                report["errors"].append({"pid": snap.pid, "tid": snap.tid,
                                                         "error": f"{type(error).__name__}: {error}"})
                        except IdentityChanged as error:
                            if mode == "apply":
                                raise
                            report["skipped"].append({"pid": snap.pid, "tid": snap.tid,
                                                      "reason": str(error)})
                        except Exception as error:
                            if mode == "apply":
                                raise
                            report["errors"].append({"pid": snap.pid, "tid": snap.tid,
                                                     "error": f"{type(error).__name__}: {error}"})
                    after = self._scan(process)
                    after_all.extend(asdict(snap) for snap in after)
                    if {_key(snap) for snap in before} != {_key(snap) for snap in after}:
                        stable = False
                    for snap in after:
                        if reused_thread(process, snap):
                            continue
                        entry = self._remember(process, snap)
                        desired = process["target_cpus"] if mode == "apply" else entry["restore_cpus"]
                        if snap.affinity != tuple(desired):
                            stable = False
                except (Exception, KeyboardInterrupt) as error:
                    if mode == "apply":
                        raise
                    if tolerate_process_exit and (_gone(error) or isinstance(error, IdentityChanged)):
                        report["skipped"].append({"pid": process["pid"], "reason": str(error)})
                        excluded.add(process["pid"])
                    else:
                        report["errors"].append({"pid": process["pid"],
                                                 "error": f"{type(error).__name__}: {error}"})
                        # Other target processes still receive restoration.
                        excluded.add(process["pid"])
            report["passes"] = pass_index + 1
            report["after"] = after_all
            if stable:
                report["converged"] = not report["errors"]
                break
        if not report["converged"] and not report["errors"]:
            report["errors"].append({"error": f"Thread masks did not converge in {self.max_passes} passes"})
        return report

    def apply(self, plans: Iterable[ProcessPlan], *, available_cpus=None) -> dict:
        if self.status != "new":
            raise ValueError("Use a new session for each apply")
        plans = tuple(plans)
        if not plans or len({plan.pid for plan in plans}) != len(plans):
            raise ValueError("Supply a nonempty set of distinct explicit PIDs")
        if available_cpus is not None:
            available = set(cpu_set(available_cpus))
            for plan in plans:
                if not set(plan.cpus) <= available or not set(plan.newborn_restore_cpus) <= available:
                    raise ValueError(f"PID {plan.pid}: target or restoration mask contains unavailable CPUs")
        # Capture every baseline before changing any target. An early capture
        # failure therefore requires no rollback and leaves all masks untouched.
        for plan in plans:
            starttime = self.backend.process_starttime(plan.pid)
            if plan.expected_starttime is not None and plan.expected_starttime != starttime:
                raise IdentityChanged(f"PID {plan.pid} does not match its expected identity")
            process = {"pid": plan.pid, "pid_starttime": starttime,
                       "target_cpus": list(plan.cpus), "newborn_restore_cpus": list(plan.newborn_restore_cpus),
                       "threads": {}}
            for snap in self._scan(process):
                self._remember(process, snap, baseline=True)
            self.processes.append(process)
        self.status = "applying"
        try:
            self._persist()
            report = self._converge("apply")
            if not report["converged"]:
                raise RuntimeError(report["errors"])
            self.status = "applied"
            self._persist()
            report["journal"] = self.to_dict()
            return report
        except BaseException as error:
            rollback = self.restore()
            raise PartitionError(f"Apply failed: {type(error).__name__}: {error}",
                                 {"error": str(error), "rollback": rollback,
                                  "journal": self.to_dict()}) from error

    def restore(self) -> dict:
        """Restore surviving identities, including births after the baseline."""
        self.status = "restoring"
        # Persistence failure must never prevent an attempted best-effort restore.
        persistence_errors = []
        original_persist = self.persist

        def safe_persist(value):
            if original_persist is not None:
                try:
                    original_persist(value)
                except Exception as error:
                    persistence_errors.append(str(error))

        self.persist = safe_persist
        try:
            report = self._converge("restore", tolerate_process_exit=True)
            self.status = "restored" if report["converged"] else "restore_incomplete"
            self._persist()
        finally:
            self.persist = original_persist
        if persistence_errors:
            report["persistence_errors"] = persistence_errors
        report["journal"] = self.to_dict()
        return report

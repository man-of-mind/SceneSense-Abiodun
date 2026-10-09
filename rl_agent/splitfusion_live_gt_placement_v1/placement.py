"""Observe and restore only the dedicated GT child's explicitly owned TIDs.

The archive must already be READY under the original mask. Neither this
module nor the client calls sched_setaffinity for any other process. Request
observations start before snapshot unpack and end after computation; Linux
schedstat supplies runnable wait, and RUSAGE_THREAD supplies context switches.
"""
import ctypes
import os
from pathlib import Path
import resource
import threading
import time

from .contract import SCHEMA

_libc = ctypes.CDLL(None, use_errno=True)
_libc.sched_getcpu.argtypes = []
_libc.sched_getcpu.restype = ctypes.c_int


def require(value, message):
    if not value:
        raise ValueError(message)


def cpu_now():
    cpu = _libc.sched_getcpu()
    if cpu < 0:
        raise OSError(ctypes.get_errno(), "sched_getcpu failed")
    return cpu


def scheduling(tid=0):
    return dict(affinity=sorted(os.sched_getaffinity(tid)),
                scheduler=os.sched_getscheduler(tid), nice=os.getpriority(os.PRIO_PROCESS, tid))


def process_threads(pid):
    """Read both kernel affinity and /proc affinity for every observed TID."""
    rows = {}
    for path in sorted(Path(f"/proc/{pid}/task").iterdir(), key=lambda p: int(p.name)):
        tid = int(path.name)
        status = {}
        for line in (path / "status").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        rows[str(tid)] = dict(scheduling(tid), proc_allowed_list=status["Cpus_allowed_list"],
                              name=status["Name"], parent_pid=int(status["PPid"]))
        values = [int(value) for value in (path / "schedstat").read_text().split()]
        require(len(values) == 3, "malformed observed thread schedstat")
        rows[str(tid)]["schedstat"] = dict(zip(("run_ns", "wait_ns", "timeslices"), values))
    return rows


def sample():
    tid = threading.get_native_id()
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    values = [int(value) for value in Path(f"/proc/self/task/{tid}/schedstat").read_text().split()]
    require(len(values) == 3, "malformed Linux thread schedstat")
    cpu = cpu_now()
    try:
        frequency = int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text())
    except (OSError, ValueError):
        frequency = None
    return dict(cpu=cpu, native_tid=tid, affinity=sorted(os.sched_getaffinity(tid)),
                raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                thread_cpu_ns=time.thread_time_ns(),
                schedstat=dict(zip(("run_ns", "wait_ns", "timeslices"), values)),
                voluntary_switches=usage.ru_nvcsw, involuntary_switches=usage.ru_nivcsw,
                rusage_thread_user_cpu_ns=round(usage.ru_utime * 1_000_000_000),
                rusage_thread_system_cpu_ns=round(usage.ru_stime * 1_000_000_000),
                scaling_cur_freq_khz=frequency)


def resource_snapshot():
    result = {}
    for name, kind in (("self", resource.RUSAGE_SELF), ("children", resource.RUSAGE_CHILDREN)):
        usage = resource.getrusage(kind)
        result[name] = dict(user_cpu_ns=round(usage.ru_utime * 1_000_000_000),
            system_cpu_ns=round(usage.ru_stime * 1_000_000_000), peak_rss_bytes=usage.ru_maxrss * 1024,
            voluntary_switches=usage.ru_nvcsw, involuntary_switches=usage.ru_nivcsw)
    return result


class Placement:
    """Separate allocation/start so partial affinity changes remain owned."""
    def __init__(self, *, requested_cpu, original_context, archive_pid, max_frames,
                 native_pool_startup=None):
        require(type(requested_cpu) is int and requested_cpu >= 0, "integer placement CPU required")
        self.pid, self.main_tid = os.getpid(), threading.get_native_id()
        require(self.pid == self.main_tid, "dedicated GT main thread required")
        require(original_context == scheduling(), "GT pre-placement scheduling differs from original parent")
        require(requested_cpu in original_context["affinity"], "placement CPU outside original allowed mask")
        self.requested_cpu, self.original_scheduling = requested_cpu, dict(original_context)
        self.active_scheduling = dict(original_context, affinity=[requested_cpu])
        self.archive_pid, self.max_frames = archive_pid, max_frames
        self.native_pool_startup = native_pool_startup
        self.original_threads = process_threads(self.pid)
        self.later_threads = {}
        self.active_threads = None
        self.archive_start = self.archive_end = None
        self.closed = self.restored = self.started = False
        self.set_calls, self.restore_calls, self.errors = [], [], []
        self.request_samples, self.pending = [], None
        self.restored_threads = None

    def _same_owner(self):
        require(os.getpid() == self.pid and threading.get_native_id() == self.main_tid,
                "placement must run on its dedicated GT main thread")

    def _check_threads(self, rows, expected):
        require(bool(rows), "empty thread context")
        for tid, row in rows.items():
            require(all(row[key] == value for key, value in expected.items()),
                    f"thread {tid} ({row.get('name')}) scheduling differs from registered placement: "
                    f"observed={ {key: row[key] for key in expected} }, expected={expected}")

    def observe_archive(self):
        rows = process_threads(self.archive_pid)
        self._check_threads(rows, self.original_scheduling)
        require(all(row["parent_pid"] == self.pid for row in rows.values()),
                "archive process is not owned by this GT child")
        # The sealed archiver is stdlib-only and may not acquire descendants.
        for tid in rows:
            require(not Path(f"/proc/{self.archive_pid}/task/{tid}/children").read_text().strip(),
                    "unexpected archive descendant")
        return dict(pid=self.archive_pid, threads=rows,
                    observed_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))

    def start(self):
        self._same_owner()
        require(not self.started and not self.closed, "placement already started/closed")
        self._check_threads(self.original_threads, self.original_scheduling)
        self.archive_start = self.observe_archive()
        try:
            # Record ownership before the syscall, including injected failures
            # occurring after the actual kernel change.
            for tid in self.original_threads:
                self.set_calls.append(int(tid))
                os.sched_setaffinity(int(tid), {self.requested_cpu})
            self.started = True
            self.verify()
            self.active_threads = process_threads(self.pid)
        except BaseException:
            self.restore()
            raise

    def verify(self):
        self._same_owner()
        require(self.started and not self.closed, "placement is not active")
        rows = process_threads(self.pid)
        # Native helpers may be lazy-created by the unchanged GT libraries.
        # Linux inherits the creating TID's singleton affinity. We never pin
        # these later helpers again; reject any whose observed mask differs.
        for tid, row in rows.items():
            if tid not in self.original_threads and tid not in self.later_threads:
                self.later_threads[tid] = dict(row)
        self._check_threads(rows, self.active_scheduling)
        require(cpu_now() == self.requested_cpu, "GT is executing outside selected CPU")
        children = set()
        for tid in rows:
            children.update(int(pid) for pid in Path(f"/proc/self/task/{tid}/children").read_text().split())
        require(children == {self.archive_pid}, "unexpected GT child process outside archive ownership")
        self.archive_end = self.observe_archive()
        return rows

    def begin_request(self):
        threads = self.verify()
        require(self.pending is None and len(self.request_samples) < self.max_frames,
                "placement request scope is overlapping/exhausted")
        self.pending = dict(request_index=len(self.request_samples), frame_id=None,
                            threads_before=threads, before=sample(),
                            interval="snapshot_unpack_to_postprocess")

    def identify_request(self, identity):
        require(self.pending is not None, "snapshot request scope absent")
        self.pending["frame_id"] = identity["frame_id"]

    def end_request(self, outcome):
        if self.pending is None:
            return
        after = sample()
        before = self.pending["before"]
        delta = dict(wall_ns=after["raw_ns"]-before["raw_ns"],
                     thread_cpu_ns=after["thread_cpu_ns"]-before["thread_cpu_ns"])
        for key in ("run_ns", "wait_ns", "timeslices"):
            delta[key] = after["schedstat"][key]-before["schedstat"][key]
        for key in ("voluntary_switches", "involuntary_switches",
                    "rusage_thread_user_cpu_ns", "rusage_thread_system_cpu_ns"):
            delta[key] = after[key]-before[key]
        row = dict(self.pending, after=after, delta=delta, outcome=outcome,
                   threads_after=process_threads(self.pid))
        self.request_samples.append(row)
        self.pending = None
        require(before["cpu"] == after["cpu"] == self.requested_cpu,
                "request CPU execution differs from selected CPU")
        require(all(value >= 0 for value in delta.values()), "request scheduler counter regressed")

    def before_drain(self):
        self._same_owner()
        if self.started and not self.closed:
            self.end_request("FAILED_OR_CLOSED")
            self.verify()

    def restore(self):
        self._same_owner()
        if self.closed:
            return
        failures = []
        try:
            # A newly created helper is forbidden at request boundaries, but
            # must also be returned to the original mask on the failure path.
            for tid in process_threads(self.pid):
                try:
                    self.restore_calls.append(int(tid))
                    os.sched_setaffinity(int(tid), set(self.original_scheduling["affinity"]))
                except ProcessLookupError:
                    continue
                except BaseException as error:
                    failures.append(repr(error))
            self.restored_threads = process_threads(self.pid)
            self._check_threads(self.restored_threads, self.original_scheduling)
            self.restored = not failures
        finally:
            self.closed = True
            self.errors.extend(failures)
        require(self.restored, "GT affinity restoration failed: " + "; ".join(failures))

    def snapshot(self, *, include_requests=False):
        self._same_owner()
        return dict(schema=SCHEMA, mode="cpu_affinity", worker_pid=self.pid,
                    requested_cpu=self.requested_cpu, main_tid=self.main_tid, cpu_now=cpu_now(),
                    affinity_scope="all_gt_child_tids", original_scheduling=self.original_scheduling,
                    active_scheduling=self.active_scheduling, original_threads=self.original_threads,
                    later_threads=self.later_threads,
                    later_thread_policy="inherit_selected_mask_verified_at_request_boundaries",
                    active_threads=self.active_threads, current_threads=process_threads(self.pid),
                    archive_start=self.archive_start, archive_end=self.archive_end,
                    native_pool_startup=self.native_pool_startup,
                    archive_end_phase="last_observation_before_archive_drain",
                    restoration_phase="after_last_request_before_sealed_close_and_archive_drain",
                    closed=self.closed, restored=self.restored, started=self.started,
                    restored_threads=self.restored_threads, set_calls=list(self.set_calls),
                    restore_calls=list(self.restore_calls), errors=list(self.errors),
                    request_count=len(self.request_samples),
                    request_samples=list(self.request_samples if include_requests else self.request_samples[-1:]),
                    request_samples_are_cumulative=include_requests,
                    resource=resource_snapshot(),
                    cpu_id_provider="libc.sched_getcpu",
                    scheduler_provider="/proc/self/task/<main_tid>/schedstat",
                    context_switch_provider="resource.getrusage(RUSAGE_THREAD)",
                    numerical_provider="rl_agent.splitfusion_live_gt_handoff_v1.engine.Engine",
                    gc_provider="rl_agent.splitfusion_live_gt_optimization_v8.engine.ControlEngine",
                    parent_affinity_mutated=False, archive_affinity_mutated=False)

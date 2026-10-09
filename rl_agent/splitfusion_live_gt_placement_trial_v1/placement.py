"""Identical observation in both arms; only treatment sets main-TID affinity."""
from rl_agent.splitfusion_live_gt_main_thread_placement_v1.placement import Placement as MainPlacement
from rl_agent.splitfusion_live_gt_placement_v1.placement import (
    require, sample, process_threads, cpu_now)
from . import contract as C


class Placement(MainPlacement):
    def __init__(self, *, mode, original_context, **kwargs):
        require(type(mode) is str and mode in C.MODES, "registered trial mode required")
        self.trial_mode = mode
        # Base allocation validates ownership and original context only. The
        # control immediately records its authentic unbound target, no syscall.
        cpu = C.TREATMENT_CPU if mode == "p4" else original_context["affinity"][0]
        super().__init__(requested_cpu=cpu, original_context=original_context, **kwargs)
        if mode == "control":
            self.requested_cpu = None
            self.active_scheduling = dict(self.original_scheduling)

    def start(self):
        if self.trial_mode == "p4":
            return super().start()
        self._same_owner()
        require(not self.started and not self.closed, "control already started/closed")
        self._check_threads(self.original_threads, self.original_scheduling)
        self.archive_start = self.observe_archive()
        self.started = True
        self.active_threads = self.verify()

    def verify(self):
        if self.trial_mode == "p4":
            return super().verify()
        self._same_owner()
        require(self.started and not self.closed, "control observation is not active")
        rows = process_threads(self.pid)
        for tid, row in rows.items():
            if tid not in self.original_threads and tid not in self.later_threads:
                self.later_threads[tid] = dict(row)
        self._check_threads(rows, self.original_scheduling)
        require(cpu_now() in self.original_scheduling["affinity"], "control CPU outside original mask")
        from pathlib import Path
        children = set()
        for tid in rows:
            children.update(int(pid) for pid in Path(f"/proc/self/task/{tid}/children").read_text().split())
        require(children == {self.archive_pid}, "unexpected control child outside archive ownership")
        self.archive_end = self.observe_archive()
        return rows

    def end_request(self, outcome):
        # Same counter/read order and payload in both arms. Control may migrate
        # between original CPUs, so its two real CPU observations need not match.
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
        self.request_samples.append(dict(self.pending, after=after, delta=delta, outcome=outcome,
                                          threads_after=process_threads(self.pid)))
        self.pending = None
        allowed = [C.TREATMENT_CPU] if self.trial_mode == "p4" else self.original_scheduling["affinity"]
        require(before["cpu"] in allowed and after["cpu"] in allowed,
                "request execution differs from registered trial mode")
        require(all(value >= 0 for value in delta.values()), "request scheduler counter regressed")

    def restore(self):
        if self.trial_mode == "p4":
            return super().restore()
        self._same_owner()
        if self.closed:
            return
        try:
            self.restored_threads = process_threads(self.pid)
            self._check_threads(self.restored_threads, self.original_scheduling)
            self.restored = True
        except BaseException as error:
            self.errors.append(repr(error))
            raise
        finally:
            self.closed = True

    def snapshot(self, *, include_requests=False):
        result = super().snapshot(include_requests=include_requests)
        result.update(schema=C.SCHEMA, trial_mode=self.trial_mode,
            mode="main_tid_affinity" if self.trial_mode == "p4" else "control_unbound",
            affinity_scope="main_gt_tid_only" if self.trial_mode == "p4" else "none_control",
            matched_startup="v8_CONTROL_one_startup_collection_no_freeze",
            placement_observer_provider="rl_agent.splitfusion_live_gt_placement_trial_v1.placement")
        return result

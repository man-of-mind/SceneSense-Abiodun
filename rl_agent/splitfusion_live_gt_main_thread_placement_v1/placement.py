"""Only the main GT TID is changed; helper/archive masks remain observable."""
import os
from pathlib import Path

from rl_agent.splitfusion_live_gt_placement_v1.placement import (
    Placement as ObservedPlacement, cpu_now, process_threads, scheduling, require)
from .contract import SCHEMA


class Placement(ObservedPlacement):
    def start(self):
        self._same_owner()
        require(not self.started and not self.closed, "placement already started/closed")
        self._check_threads(self.original_threads, self.original_scheduling)
        self.archive_start = self.observe_archive()
        try:
            self.set_calls.append(self.main_tid)
            os.sched_setaffinity(self.main_tid, {self.requested_cpu})
            self.started = True
            self.active_threads = self.verify()
        except BaseException:
            self.restore()
            raise

    def verify(self):
        self._same_owner()
        require(self.started and not self.closed, "placement is not active")
        rows = process_threads(self.pid)
        for tid, row in rows.items():
            if tid not in self.original_threads and tid not in self.later_threads:
                self.later_threads[tid] = dict(row)
            expected = self.active_scheduling if int(tid) == self.main_tid else self.original_scheduling
            self._check_threads({tid: row}, expected)
        require(str(self.main_tid) in rows and cpu_now() == self.requested_cpu,
                "main GT TID is not executing on its selected CPU")
        children = set()
        for tid in rows:
            children.update(int(pid) for pid in Path(f"/proc/self/task/{tid}/children").read_text().split())
        require(children == {self.archive_pid}, "unexpected GT child outside archive ownership")
        self.archive_end = self.observe_archive()
        return rows

    def restore(self):
        self._same_owner()
        if self.closed:
            return
        try:
            self.restore_calls.append(self.main_tid)
            os.sched_setaffinity(self.main_tid, set(self.original_scheduling["affinity"]))
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
        result.pop("native_pool_startup")
        result.update(schema=SCHEMA, mode="main_tid_affinity", affinity_scope="main_gt_tid_only",
            later_thread_policy="helpers_must_preserve_original_mask",
            helper_affinity_mutated=False, native_pool_preparation=False,
            placement_observer_provider="rl_agent.splitfusion_live_gt_placement_v1.placement")
        return result

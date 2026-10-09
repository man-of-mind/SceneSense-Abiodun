"""Mocked affinity/identity races plus a disposable subprocess inheritance check."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from cpu_partition import (IdentityChanged, PartitionError, PartitionSession,
                           ProcessPlan, ProcBackend, TaskSnapshot, cpu_set,
                           validate_disjoint_cpu_sets)


class FakeBackend:
    def __init__(self):
        self.processes = {}
        self.calls = []
        self.fail_once_tid = None
        self.always_fail_tid = None
        self.after_write = None
        self.on_list = None

    def add_process(self, pid, *, starttime=100, cpus=(0, 1, 2, 3), tids=()):
        self.processes[pid] = {"starttime": starttime, "tasks": {}}
        self.add_thread(pid, pid, starttime=starttime, cpus=cpus)
        for tid in tids:
            self.add_thread(pid, tid, cpus=cpus)

    def add_thread(self, pid, tid, *, starttime=200, cpus=(0, 1, 2, 3)):
        self.processes[pid]["tasks"][tid] = TaskSnapshot(
            pid, self.processes[pid]["starttime"], tid, starttime,
            f"worker ({tid})", tuple(cpus), 0, 0)

    def process_starttime(self, pid):
        if pid not in self.processes:
            raise ProcessLookupError("process exited")
        return self.processes[pid]["starttime"]

    def list_tids(self, pid):
        if self.on_list:
            self.on_list(self, pid)
        return sorted(self.processes[pid]["tasks"])

    def snapshot(self, pid, tid):
        self.process_starttime(pid)
        if tid not in self.processes[pid]["tasks"]:
            raise ProcessLookupError("thread exited")
        return self.processes[pid]["tasks"][tid]

    def set_affinity(self, tid, cpus):
        self.calls.append((tid, tuple(cpus)))
        if self.fail_once_tid == tid:
            self.fail_once_tid = None
            raise PermissionError("injected one-shot write failure")
        if self.always_fail_tid == tid:
            raise PermissionError("injected permanent write failure")
        for process in self.processes.values():
            if tid in process["tasks"]:
                process["tasks"][tid] = replace(process["tasks"][tid], affinity=tuple(cpus))
                if self.after_write:
                    self.after_write(self, tid)
                return
        raise ProcessLookupError("thread exited")


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.backend.add_process(10, tids=(11, 12))
        self.backend.add_process(20, tids=(21,))

    def session(self, **kwargs):
        return PartitionSession(self.backend, **kwargs)

    def plan(self, pid=10, cpus=(2, 3)):
        return ProcessPlan(pid, cpus, (0, 1, 2, 3))

    def test_all_existing_threads_only_explicit_process_and_priorities_preserved(self):
        self.backend.processes[10]["tasks"][11] = replace(
            self.backend.snapshot(10, 11), policy=1, priority=42)
        report = self.session().apply([self.plan()])
        self.assertTrue(report["converged"])
        self.assertEqual({tid for tid, _ in self.backend.calls}, {10, 11, 12})
        for tid in (10, 11, 12):
            self.assertEqual(self.backend.snapshot(10, tid).affinity, (2, 3))
        self.assertEqual(self.backend.snapshot(10, 11).priority, 42)
        self.assertEqual(self.backend.snapshot(10, 11).policy, 1)
        self.assertEqual(self.backend.snapshot(20, 20).affinity, (0, 1, 2, 3))
        self.assertEqual(len(report["before"]), 3)
        self.assertEqual(len(report["after"]), 3)

    def test_newborn_inherits_modified_parent_and_restores_explicit_fallback(self):
        def birth(backend, tid):
            if tid == 10:
                backend.after_write = None
                backend.add_thread(10, 13, cpus=backend.snapshot(10, 10).affinity)
        self.backend.after_write = birth
        session = self.session()
        report = session.apply([self.plan()])
        self.assertGreaterEqual(report["passes"], 2)
        self.assertEqual(self.backend.snapshot(10, 13).affinity, (2, 3))
        newborn = session.to_dict()["processes"][0]["threads"]["13:200"]
        self.assertFalse(newborn["baseline"])
        self.assertEqual(newborn["restore_cpus"], [0, 1, 2, 3])
        restore = session.restore()
        self.assertTrue(restore["converged"])
        self.assertEqual(self.backend.snapshot(10, 13).affinity, (0, 1, 2, 3))

    def test_birth_after_apply_is_discovered_during_restoration(self):
        session = self.session()
        session.apply([self.plan()])
        self.backend.add_thread(10, 13, cpus=(2, 3))
        session.restore()
        self.assertEqual(self.backend.snapshot(10, 13).affinity, (0, 1, 2, 3))

    def test_newborn_with_wide_mask_is_restricted_on_next_pass(self):
        def birth(backend, tid):
            if tid == 10:
                backend.after_write = None
                backend.add_thread(10, 13)
        self.backend.after_write = birth
        report = self.session().apply([self.plan()])
        self.assertGreaterEqual(report["passes"], 2)
        self.assertEqual(self.backend.snapshot(10, 13).affinity, (2, 3))

    def test_heterogeneous_original_masks_restore_exactly(self):
        self.backend.processes[10]["tasks"][11] = replace(
            self.backend.snapshot(10, 11), affinity=(1, 2))
        session = self.session()
        session.apply([self.plan()])
        session.restore()
        self.assertEqual(self.backend.snapshot(10, 11).affinity, (1, 2))
        self.assertEqual(self.backend.snapshot(10, 12).affinity, (0, 1, 2, 3))

    def test_pid_reuse_blocks_apply_without_writes(self):
        with self.assertRaises(IdentityChanged):
            self.session().apply([ProcessPlan(10, (2, 3), (0, 1, 2, 3), 999)])
        self.assertFalse(self.backend.calls)

    def test_pid_reuse_is_skipped_during_restore(self):
        session = self.session()
        session.apply([self.plan()])
        self.backend.add_process(10, starttime=999, cpus=(1,))
        self.backend.calls.clear()
        report = session.restore()
        self.assertTrue(report["skipped"])
        self.assertFalse(self.backend.calls)
        self.assertEqual(self.backend.snapshot(10, 10).affinity, (1,))

    def test_pid_reuse_during_journal_io_blocks_write_and_rollback(self):
        def reuse_on_pending_journal(value):
            if value["events"]:
                self.backend.add_process(10, starttime=999, cpus=(1,))
        session = self.session(persist=reuse_on_pending_journal)
        with self.assertRaises(PartitionError) as caught:
            session.apply([self.plan()])
        self.assertTrue(caught.exception.report["rollback"]["skipped"])
        self.assertFalse(self.backend.calls)
        self.assertEqual(self.backend.snapshot(10, 10).affinity, (1,))

    def test_exited_process_is_skipped_during_restore(self):
        session = self.session()
        session.apply([self.plan(), self.plan(20)])
        del self.backend.processes[10]
        report = session.restore()
        self.assertTrue(report["converged"])
        self.assertTrue(report["skipped"])
        self.assertEqual(self.backend.snapshot(20, 21).affinity, (0, 1, 2, 3))

    def test_birth_during_restore_converges_to_fallback(self):
        session = self.session()
        session.apply([self.plan()])
        def birth(backend, tid):
            if tid == 10:
                backend.after_write = None
                backend.add_thread(10, 13, cpus=(2, 3))
        self.backend.after_write = birth
        report = session.restore()
        self.assertGreaterEqual(report["passes"], 2)
        self.assertTrue(report["converged"])
        self.assertEqual(self.backend.snapshot(10, 13).affinity, (0, 1, 2, 3))

    def test_tid_reuse_is_skipped_while_other_threads_restore(self):
        session = self.session()
        session.apply([self.plan()])
        self.backend.add_thread(10, 11, starttime=999, cpus=(1,))
        self.backend.calls.clear()
        report = session.restore()
        self.assertTrue(report["skipped"])
        self.assertNotIn(11, [tid for tid, _ in self.backend.calls])
        self.assertEqual(self.backend.snapshot(10, 11).affinity, (1,))
        self.assertEqual(self.backend.snapshot(10, 12).affinity, (0, 1, 2, 3))

    def test_rollback_after_partial_failure(self):
        self.backend.fail_once_tid = 12
        with self.assertRaises(PartitionError) as caught:
            self.session().apply([self.plan(), self.plan(20)])
        self.assertTrue(caught.exception.report["rollback"]["converged"])
        for pid in (10, 20):
            for tid in self.backend.list_tids(pid):
                self.assertEqual(self.backend.snapshot(pid, tid).affinity, (0, 1, 2, 3))

    def test_rollback_keeps_restoring_after_one_thread_failure(self):
        session = self.session()
        session.apply([self.plan(), self.plan(20)])
        self.backend.always_fail_tid = 11
        report = session.restore()
        self.assertFalse(report["converged"])
        self.assertTrue(report["errors"])
        self.assertEqual(self.backend.snapshot(10, 12).affinity, (0, 1, 2, 3))
        self.assertEqual(self.backend.snapshot(20, 21).affinity, (0, 1, 2, 3))

    def test_journal_roundtrip_restores_in_separate_session(self):
        journals = []
        session = self.session(persist=journals.append)
        session.apply([self.plan()])
        self.assertEqual(journals[1]["events"][-1]["outcome"], "pending")
        value = json.loads(json.dumps(session.to_dict()))
        restored = PartitionSession.from_dict(value, self.backend).restore()
        self.assertTrue(restored["converged"])
        self.assertEqual(self.backend.snapshot(10, 11).affinity, (0, 1, 2, 3))

    def test_persistence_failure_rolls_back_and_reports_journal_failure(self):
        calls = []
        def failing_persist(value):
            calls.append(value)
            if len(calls) >= 3:
                raise OSError("journal volume unavailable")
        session = self.session(persist=failing_persist)
        with self.assertRaises(PartitionError) as caught:
            session.apply([self.plan()])
        self.assertTrue(caught.exception.report["rollback"]["converged"])
        self.assertTrue(caught.exception.report["rollback"]["persistence_errors"])
        for tid in (10, 11, 12):
            self.assertEqual(self.backend.snapshot(10, tid).affinity, (0, 1, 2, 3))

    def test_bounded_convergence_failure_rolls_back(self):
        state = {"next_tid": 30}
        def births(backend, pid):
            if len(backend.calls) and len(backend.calls) < 10:
                state["next_tid"] += 1
                backend.add_thread(pid, state["next_tid"])
        self.backend.on_list = births
        with self.assertRaises(PartitionError) as caught:
            self.session(max_passes=1).apply([self.plan()])
        self.assertIn("did not converge", str(caught.exception))

    def test_validation_no_writes(self):
        for cpus in ((), (-1,), (True,), (1.5,)):
            with self.subTest(cpus=cpus), self.assertRaises(ValueError):
                cpu_set(cpus)
        with self.assertRaises(ValueError):
            ProcessPlan(10, (1,), ())
        with self.assertRaises(ValueError):
            self.session().apply([self.plan()], available_cpus=(0, 1))
        with self.assertRaises(ValueError):
            self.session().apply([self.plan(), self.plan()])
        with self.assertRaises(ValueError):
            self.session().apply([])
        self.assertFalse(self.backend.calls)

    def test_disjoint_topology_validation_is_separate_from_pid_masks(self):
        self.assertEqual(validate_disjoint_cpu_sets({"edge": (2, 3), "radio": (0, 1)},
                                                   range(4))["edge"], (2, 3))
        for masks in ({"edge": (), "radio": (0,)}, {"edge": (2,), "radio": (2,)},
                      {"edge": (5,), "radio": (0,)}):
            with self.subTest(masks=masks), self.assertRaises(ValueError):
                validate_disjoint_cpu_sets(masks, range(4))
        self.assertTrue(self.session().apply([self.plan(), self.plan(20)])["converged"])

    def test_proc_stat_handles_parentheses_and_spaces(self):
        value = "123 (thread ) ( name) " + " ".join(["R"] + ["0"] * 18 + ["456"] + ["0"] * 30)
        self.assertEqual(ProcBackend.parse_stat(value), (456, "thread ) ( name"))

    def test_real_disposable_subprocess_thread_inheritance(self):
        if not hasattr(os, "sched_setaffinity") or not Path("/proc/self/task").exists():
            self.skipTest("Linux affinity interface required")
        # Only this short-lived CHILD process's masks are changed. Existing
        # workload PIDs and the unittest parent's affinity remain untouched.
        code = r'''
import os, threading
from cpu_partition import PartitionSession, ProcessPlan
original = tuple(sorted(os.sched_getaffinity(0)))
session = PartitionSession()
session.apply([ProcessPlan(os.getpid(), original[:1], original)])
ready, finish = threading.Event(), threading.Event()
result = {}
def worker():
    result["tid"] = threading.get_native_id()
    result["mask"] = tuple(sorted(os.sched_getaffinity(0)))
    ready.set()
    finish.wait(5)
t = threading.Thread(target=worker)
t.start()
try:
    assert ready.wait(5)
    assert result["mask"] == original[:1], result
    report = session.restore()
    assert report["converged"], report
    assert tuple(sorted(os.sched_getaffinity(result["tid"]))) == original
    assert tuple(sorted(os.sched_getaffinity(0))) == original
finally:
    finish.set()
    t.join(5)
'''
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=str(Path(__file__).parent), capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

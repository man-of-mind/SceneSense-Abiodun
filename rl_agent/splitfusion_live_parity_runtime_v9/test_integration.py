"""Actual two-arm worker/session/archive seams, no live services or GPU."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from rl_agent.splitfusion_live_parity_runtime_v8 import test_integration as V8
from rl_agent.splitfusion_live_gt_optimization_v7 import test_lifecycle as L7
from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as T
from . import contract as C, entries as E, runtime as R, worker_observer as O
from . import host_capture, placement_evidence as P, process_scope as S
from .bindings import function

_worker_test = function(V8.WorkerIntegrationTests.test_two_real_frames_caps_archive_identity_and_genuine_close,
    dict(vars(V8), E=E, C=C, O=O, H=C.WORKER, HR=R, __package__=__package__, __spec__=None))


class _PrivateEntry:
    """Fixture patches only this adapter's private retained namespace."""
    def __getattr__(self, name):
        try:
            return E._bound[name]
        except KeyError as error:
            raise AttributeError(name) from error
    def __setattr__(self, name, value):
        E._bound[name] = value
    def __delattr__(self, name):
        del E._bound[name]


class CombinedTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with V8.WorkerIntegrationTests.fixture(self) as f:
            start = f.start_job
            observed = []
            def start_job():
                if not observed:
                    client = f.observation["state"]["clients"][0]
                    for pid, ppid in ((client.process.pid, os.getpid()),
                                      (client.archive_startup["pid"], client.process.pid)):
                        proc = Path("/proc") / str(pid)
                        values = dict(item.split("=", 1) for item in
                            (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
                        observed.append(dict(pid=pid, ppid=ppid,
                            cmdline=(proc / "cmdline").read_bytes().decode().replace("\0", " ").strip(),
                            variables={key: values.get(key) for key in S.V7.ENVIRONMENT}, role=None,
                            descendant_of_runner=True, in_container=False, environ_source="direct"))
                return start()
            f.start_job = start_job
            yield f
            proof = P.verify_trial_scope(f.attempt, mode=self.mode)
            self.assertTrue(proof["passed"])
            timeline = json.loads((f.attempt / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
            scope = json.loads((f.attempt / "GT_WORKER_SCOPE.json").read_bytes())
            # The sensor/UE owner is a fixture; child PIDs, parent relationships,
            # command lines, cap observations and every native receipt are real.
            baseline = dict({key: os.environ.get(key) for key in S.V7.ENVIRONMENT},
                            OMP_NUM_THREADS=None, MKL_NUM_THREADS=None)
            parent = dict(pid=os.getpid(), ppid=os.getppid(), role="ue",
                variables=dict(baseline, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                cmdline="/usr/bin/python3 -m " + C.CHILD, environ_source="direct",
                descendant_of_runner=True, in_container=False)
            rows = [parent, *observed]
            registration = dict(worker_sources=V8.W.sources(), bootstrap_sources=T.sources(),
                handoff_binding=C.WORKER.binding(), worker_client_factory=O.FACTORY,
                transmitted_budget=C.BUDGET, publisher_uid=os.getuid(), publisher_gid=os.getgid(),
                child_module=C.CHILD)
            children = S.verify_registered_children(registration, scope, timeline, rows,
                baseline=baseline, expected_worker_sources=V8.W.sources(),
                expected_bootstrap_sources=T.sources(), expected_binding=C.WORKER.binding(),
                expected_factory=O.FACTORY, expected_bootstrap_module=T.BOOTSTRAP)
            self.assertEqual(set(children["roles"]), {str(row["pid"]) for row in observed})
            self.assertTrue(S.classify_inventory(rows, baseline=baseline, children=children)
                            ["observed_other_processes_baseline"])
            foreign = dict(observed[1], pid=99999999,
                variables=dict(observed[1]["variables"], OMP_NUM_THREADS="2"))
            foreign_rows = [*rows, foreign]
            foreign_children = S.verify_registered_children(registration, scope, timeline, foreign_rows,
                baseline=baseline, expected_worker_sources=V8.W.sources(),
                expected_bootstrap_sources=T.sources(), expected_binding=C.WORKER.binding(),
                expected_factory=O.FACTORY, expected_bootstrap_module=T.BOOTSTRAP)
            self.assertFalse(S.classify_inventory(foreign_rows, baseline=baseline, children=foreign_children)
                             ["observed_other_processes_baseline"])
            altered = copy.deepcopy(timeline["clients"][0]["ready"]["process_context"])
            altered["threads"]["999999"] = dict(altered["scheduling"], affinity=[4])
            if self.mode == "p4":
                with self.assertRaises(ValueError):
                    P.original_scheduling_reference(altered, altered["pid"], mode=self.mode)

    def run_mode(self, mode):
        self.mode = mode
        with mock.patch.object(host_capture, "verify_prospective", return_value={"trial": C.trial(mode)}):
            _worker_test(self)

    def test_control_two_frames_actual_session_and_exact_process_scope(self):
        self.run_mode("control")

    def test_p4_two_frames_actual_session_restoration_and_exact_process_scope(self):
        self.run_mode("p4")

    def test_source_wire_identity_and_retained_globals_unchanged(self):
        self.assertNotEqual(T.sources(), V8.H.sources())
        self.assertEqual(C.handoff_binding(Path(__file__).resolve().parents[2]), V8.H.binding())
        self.assertIs(V8.E.C, V8.C)
        self.assertIs(E.edge.__globals__["dispatch_with_retained_shutdown"], L7.L.dispatch_with_retained_shutdown)
        self.assertIn("dispatch_with_retained_shutdown", E.edge.__code__.co_names)


class RetainedShutdownSeamTests(L7.LifecycleTests):
    """All real scored retained-shutdown adversarial cases use the live adapter."""


class EdgeEntryTests(unittest.TestCase):
    test_actual_entry_scores_and_closes = function(
        V8.EdgeEntryIntegrationTests.test_actual_entry_scores_packet_and_preserves_full_close_after_broker_exit,
        dict(vars(V8), E=_PrivateEntry(), C=C, H=C.WORKER, __package__=__package__, __spec__=None))


if __name__ == "__main__":
    unittest.main()

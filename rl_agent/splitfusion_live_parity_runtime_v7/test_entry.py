"""Fast actual-entry ordering checks; real worker composition is tested separately."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

from . import entries as E, contract as C, optimization_scope as O, startup as S
from . import worker_observer as W
from rl_agent.splitfusion_paired_quality_replay_v1 import capture as A, scope as I


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    result.__path__ = []
    return result


class EntryTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, failure=None, front_alive=False, gt_alive=False,
                timeline_complete=True, archive_fault=False):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = Path(tmp)
            attempt = output / "capture/cells/cell"
            attempt.mkdir(parents=True)
            record = dict(budget=C.BUDGET, optimization_sources={"runtime.py": "a"*64},
                worker_sources={"rl_agent/worker.py": "b"*64}, worker_timeout_s=30.,
                worker_client_factory=W.FACTORY)
            (output / "PROSPECTIVE.json").write_text(json.dumps(record))
            config = {"qualification_id": C.ID}
            config["sha256"] = C.canonical(config)
            events = []
            collectors = [NS(worker=NS(is_alive=lambda: front_alive),
                evaluation_worker=NS(is_alive=lambda: gt_alive))]
            owned = dict(collectors=collectors, manifest=None,
                inspect_drain=lambda: {"worker_drained": not gt_alive})
            optimized = dict(sources=record["optimization_sources"], settings_before={"fixed": True})
            worker = dict(schema="splitfusion.cpu_gt_worker.runtime_scope.v4",
                sources=record["worker_sources"], complete=not gt_alive, errors=[],
                joint_drained=not gt_alive, worker_close_succeeded=True,
                preserved_before_teardown=True, new_cpu_processes=1, new_queues=0,
                new_python_threads=0, no_cuda_work=True)
            client_factory = object()

            @contextmanager
            def scope(name, value=None, **kwargs):
                events.append("enter:" + name)
                try:
                    yield value
                finally:
                    events.append("exit:" + name)
                    if name == "worker":
                        (attempt / "GT_WORKER_SCOPE.json").write_text(json.dumps(worker))
                    if name == "observer":
                        (attempt / "CPU_GT_PROCESS_TIMELINE.json").write_text(json.dumps(
                            dict(sources=record["worker_sources"], complete=timeline_complete,
                                dropped=0, errors=[])))
                    if name == "ownership":
                        self.assertIn("exit:worker", events)
                        self.assertIn("exit:observer", events)
                        owned["manifest"] = {"complete": not gt_alive}
                    if name == "optimization":
                        optimized.update(settings_after={"fixed": True},
                            retained_settings_unchanged=True, source_integrity_error=None)

            def worker_scope(**kwargs):
                self.assertIs(kwargs["_client_factory"], client_factory)
                self.assertEqual(kwargs["expected_sources"], record["worker_sources"])
                self.assertEqual(kwargs["timeout_s"], 30.)
                return scope("worker", worker)

            class Archive:
                errors = ["injected"] if archive_fault else []
                error_count = 1 if archive_fault else 0
                policy_opportunities = 1
                def __init__(self, **kwargs):
                    if kwargs != dict(max_frames=C.INPUT_MAX_FRAMES, max_bytes=C.INPUT_MAX_BYTES):
                        raise AssertionError(kwargs)
                def flush(self, path, *, collectors, source_binding):
                    events.append("flush")
                    expected = ["exit:worker", "exit:observer", "exit:ownership",
                        "exit:optimization", "exit:capture", "flush"]
                    positions = [events.index(item) for item in expected]
                    if positions != sorted(positions):
                        raise AssertionError("archive flushed before joint drain / ownership exit")
                    drain = A.verified_drain(collectors)
                    if self.error_count:
                        raise ValueError("archive observer fault")
                    path.mkdir()
                    (path / "MANIFEST.json").write_text("fixture")
                    return dict(frames=[{}], retained_bytes=1, drain=drain)

            def run(args, **kwargs):
                events.append("retained")
                if failure is not None:
                    raise failure
                return 0

            timing = mock.Mock(side_effect=AssertionError("parent timing is forbidden"))
            fake = {
                "rl_agent.splitfusion_live_parity_repair_v1": module("repair", composition=NS(run_ue_child=run)),
                "rl_agent.splitfusion_single_gpu_profiler_v1": module("profiler", child_v1=NS(
                    attempt_dir_from_argv=lambda _: attempt, load_config=lambda _: config)),
                "rl_agent.splitfusion_live_parity_runtime_v1": module("prior", pose=NS(
                    install_ue_pose=lambda **kw: scope("pose"))),
                "rl_agent.splitfusion_live_parity_repair_v2.source_ownership": module("ownership",
                    install_source_ownership=lambda **kw: scope("ownership", owned)),
                "rl_agent.splitfusion_live_parity_repair_v2.gt_timing": module("timing", install_gt_timing=timing),
                "rl_agent.splitfusion_live_gt_optimization_v4.runtime": module("runtime",
                    install_cpu_gt_worker=worker_scope),
                "rl_agent.splitfusion_live_parity_runtime_v7.host_capture": module("host",
                    verify_prospective=lambda _: record),
            }
            stack.enter_context(mock.patch.dict(sys.modules, fake))
            stack.enter_context(mock.patch.object(A, "InputArchive", Archive))
            stack.enter_context(mock.patch.object(I, "install_capture", lambda _: scope("capture")))
            stack.enter_context(mock.patch.object(O, "install_cpu_gt_optimizations", lambda **kw: scope("optimization", optimized)))
            stack.enter_context(mock.patch.object(E, "evidence_scope", lambda _: scope("evidence")))
            stack.enter_context(mock.patch.object(S, "preserve_startup", lambda _: scope("startup")))
            stack.enter_context(mock.patch.object(W, "observe_cpu_worker", lambda **kw: scope("observer", {"factory": client_factory})))
            yield NS(attempt=attempt, config=config, events=events, timing=timing,
                report=lambda: json.loads((attempt / "PAIRED_INPUT_SCOPE.json").read_bytes()),
                gt_report=lambda: json.loads((attempt / "GT_OPTIMIZATION_SCOPE.json").read_bytes()))

    def test_order_joint_drain_before_ownership_and_no_parent_timing(self):
        with self.fixture() as f:
            self.assertEqual(E.child([]), 0)
            self.assertEqual(f.events[:5], ["enter:capture", "enter:optimization", "enter:ownership",
                "enter:observer", "enter:worker"])
            self.assertTrue(f.report()["complete"])
            self.assertEqual(f.gt_report()["status"], "COMPLETE")
            f.timing.assert_not_called()

    def test_undrained_gt_refused_before_archive(self):
        with self.fixture(gt_alive=True) as f:
            with self.assertRaisesRegex(RuntimeError, "did not close"):
                E.child([])
            self.assertNotIn("flush", f.events)
            self.assertEqual(f.gt_report()["status"], "FAILED")

    def test_incomplete_observer_refused_before_archive(self):
        with self.fixture(timeline_complete=False) as f:
            with self.assertRaisesRegex(RuntimeError, "did not close"):
                E.child([])
            self.assertNotIn("flush", f.events)
            self.assertFalse(f.report()["complete"])

    def test_front_worker_alive_refuses_archive(self):
        with self.fixture(front_alive=True) as f:
            with self.assertRaisesRegex(ValueError, "worker has not drained"):
                E.child([])
            self.assertFalse(f.report()["complete"])

    def test_primary_exception_preserved_and_all_scopes_exit(self):
        failure = RuntimeError("retained runtime failure")
        with self.fixture(failure=failure) as f:
            with self.assertRaises(RuntimeError) as caught:
                E.child([])
            self.assertIs(caught.exception, failure)
            self.assertNotIn("flush", f.events)
            self.assertLess(f.events.index("exit:worker"), f.events.index("exit:ownership"))
            self.assertFalse(f.report()["complete"])
            f.timing.assert_not_called()

    def test_foreign_registration_before_scopes(self):
        with self.fixture() as f:
            f.config["qualification_id"] = "foreign"
            with self.assertRaises(ValueError):
                E.child([])
            self.assertEqual(f.events, [])


if __name__ == "__main__":
    unittest.main()

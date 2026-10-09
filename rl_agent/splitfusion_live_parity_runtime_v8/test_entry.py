"""Actual v8 entry lifecycle and fail-closed registration checks."""
from contextlib import contextmanager, ExitStack
import json
import os
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
    def fixture(self, *, failure=None, front_alive=False, gt_alive=False, proof_complete=True):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = Path(tmp)
            attempt = output / "capture/cells/cell"
            attempt.mkdir(parents=True)
            record = dict(budget=C.BUDGET, optimization_sources={"runtime.py": "a"*64},
                worker_sources={"rl_agent/worker.py": "b"*64}, worker_timeout_s=30.,
                bootstrap_sources=E.H.sources(), handoff_binding=E.H.binding(),
                publisher_uid=os.getuid(), publisher_gid=os.getgid(), worker_client_factory=W.FACTORY)
            (output / "PROSPECTIVE.json").write_text(json.dumps(record))
            config = {"qualification_id": C.ID}
            config["sha256"] = C.canonical(config)
            events = []
            collectors = [NS(worker=NS(is_alive=lambda: front_alive),
                evaluation_worker=NS(is_alive=lambda: gt_alive))]
            owned = dict(collectors=collectors, manifest=None,
                inspect_drain=lambda: {"worker_drained": not gt_alive})
            optimized = dict(sources=record["optimization_sources"], settings_before={"fixed": True})
            worker, factory = {}, object()

            @contextmanager
            def scope(name, value=None, **kwargs):
                events.append("enter:" + name)
                try:
                    yield value
                finally:
                    events.append("exit:" + name)
                    if name == "ownership":
                        self.assertIn("exit:worker", events)
                        self.assertIn("exit:observer", events)
                        owned["manifest"] = {"complete": not gt_alive}
                    if name == "optimization":
                        optimized.update(settings_after={"fixed": True},
                            retained_settings_unchanged=True, source_integrity_error=None)

            def worker_scope(**kwargs):
                self.assertIs(kwargs["_client_factory"], factory)
                self.assertEqual(kwargs["expected_sources"], record["worker_sources"])
                self.assertEqual(kwargs["expected_bootstrap_sources"], record["bootstrap_sources"])
                self.assertEqual(kwargs["timeout_s"], 30.)
                return scope("worker", worker)

            class Archive:
                errors, error_count, policy_opportunities = [], 0, 1
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
                    path.mkdir()
                    (path / "MANIFEST.json").write_text("fixture")
                    return dict(frames=[{}], retained_bytes=1, drain=drain)

            def run(args, **kwargs):
                events.append("retained")
                self.assertEqual(kwargs, dict(max_frames=C.BUDGET, edge_module=C.EDGE))
                if failure is not None:
                    raise failure
                return 0

            fake = {
                "rl_agent.splitfusion_live_parity_repair_v1": module("repair", composition=NS(run_ue_child=run)),
                "rl_agent.splitfusion_single_gpu_profiler_v1": module("profiler", child_v1=NS(
                    attempt_dir_from_argv=lambda _: attempt, load_config=lambda _: config)),
                "rl_agent.splitfusion_live_parity_runtime_v1": module("prior", pose=NS(
                    install_ue_pose=lambda **kw: scope("pose"))),
                "rl_agent.splitfusion_live_parity_repair_v2.source_ownership": module("ownership",
                    install_source_ownership=lambda **kw: scope("ownership", owned)),
                "rl_agent.splitfusion_live_gt_handoff_v1.runtime": module("runtime", install_cpu_gt_worker=worker_scope),
                "rl_agent.splitfusion_live_parity_runtime_v8.host_capture": module("host", verify_prospective=lambda _: record)}
            stack.enter_context(mock.patch.dict(sys.modules, fake))
            stack.enter_context(mock.patch.object(A, "InputArchive", Archive))
            stack.enter_context(mock.patch.object(I, "install_capture", lambda _: scope("capture")))
            stack.enter_context(mock.patch.object(O, "install_cpu_gt_optimizations", lambda **kw: scope("optimization", optimized)))
            stack.enter_context(mock.patch.object(E, "evidence_scope", lambda _: scope("evidence")))
            stack.enter_context(mock.patch.object(E, "_worker_proof", return_value={"complete": proof_complete}))
            stack.enter_context(mock.patch.object(S, "preserve_startup", lambda _: scope("startup")))
            stack.enter_context(mock.patch.object(W, "observe_cpu_worker", lambda **kw: scope("observer", {"factory": factory})))
            yield NS(attempt=attempt, config=config, events=events, record=record,
                report=lambda: json.loads((attempt / "PAIRED_INPUT_SCOPE.json").read_bytes()),
                gt_report=lambda: json.loads((attempt / "GT_OPTIMIZATION_SCOPE.json").read_bytes()))

    def test_joint_drain_before_ownership_and_paired_archive(self):
        with self.fixture() as f:
            self.assertEqual(E.child([]), 0)
            self.assertEqual(f.events[:5], ["enter:capture", "enter:optimization", "enter:ownership",
                "enter:observer", "enter:worker"])
            self.assertTrue(f.report()["complete"])
            self.assertEqual(f.report()["schema"], "splitfusion.paired_input.scope.v8")
            self.assertEqual(f.gt_report()["status"], "COMPLETE")

    def test_failed_handoff_proof_refuses_paired_archive(self):
        with self.fixture(proof_complete=False) as f:
            with self.assertRaisesRegex(RuntimeError, "did not close"):
                E.child([])
            self.assertNotIn("flush", f.events)
            self.assertFalse(f.report()["complete"])
            self.assertEqual(f.gt_report()["status"], "FAILED")

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

    def test_foreign_bootstrap_before_scopes(self):
        with self.fixture() as f:
            f.record["bootstrap_sources"] = {}
            with self.assertRaises(ValueError):
                E.child([])
            self.assertEqual(f.events, [])

    def test_missing_publisher_before_scopes(self):
        with self.fixture() as f:
            del f.record["publisher_uid"]
            with self.assertRaisesRegex(ValueError, "UID/GID"):
                E.child([])
            self.assertEqual(f.events, [])


if __name__ == "__main__":
    unittest.main()

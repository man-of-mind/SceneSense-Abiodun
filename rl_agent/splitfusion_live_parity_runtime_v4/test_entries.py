"""CPU-only tests of v4 scope ordering, binding, drain and lifecycle checks."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from . import contract as C, entries, optimization_scope, startup


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    result.__path__ = []
    return result


class ChildScopeTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, failure=None, drained=True, prewarm_failure=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            output = Path(directory)
            attempt = output / "capture/cells/cell"
            attempt.mkdir(parents=True)
            record = {"budget": C.BUDGET, "optimization_sources": {"runtime.py": "a" * 64}}
            (output / "PROSPECTIVE.json").write_text(json.dumps(record, indent=2) + "\n")
            config = {"qualification_id": C.ID}
            config["sha256"] = C.canonical(config)
            events = []
            owned = {"manifest": None, "inspect_drain": lambda: {"worker_drained": drained}}
            optimized = {"sources": dict(record["optimization_sources"]), "settings_before": {"fixed": True}}

            @contextmanager
            def scope(name, value=None, **kwargs):
                events.append("enter:" + name)
                if name == "optimization":
                    self.assertEqual(kwargs, {"expected_sources": record["optimization_sources"]})
                    if prewarm_failure:
                        raise RuntimeError("prewarm failed")
                if name == "timing":
                    self.assertIs(kwargs["drain_check"](), drained)
                try:
                    yield value
                finally:
                    events.append("exit:" + name)
                    if name == "ownership":
                        owned["manifest"] = {"complete": drained}
                    elif name == "optimization":
                        optimized.update(settings_after={"fixed": True}, retained_settings_unchanged=True,
                                         source_integrity_error=None)

            def run(args, **kwargs):
                self.assertEqual(args, ["fixture"])
                self.assertEqual(kwargs, {"max_frames": C.BUDGET, "edge_module": C.EDGE})
                events.append("retained")
                if failure is not None:
                    raise failure
                return 0

            fake = {
                "rl_agent.splitfusion_live_parity_repair_v1": module("repair", composition=SimpleNamespace(run_ue_child=run)),
                "rl_agent.splitfusion_single_gpu_profiler_v1": module("profiler", child_v1=SimpleNamespace(
                    attempt_dir_from_argv=lambda _: attempt, load_config=lambda _: config)),
                "rl_agent.splitfusion_live_parity_runtime_v1": module("prior", pose=SimpleNamespace(
                    install_ue_pose=lambda **kw: scope("pose", **kw))),
                "rl_agent.splitfusion_live_parity_repair_v2.source_ownership": module("ownership",
                    install_source_ownership=lambda **kw: scope("ownership", owned, **kw)),
                "rl_agent.splitfusion_live_parity_repair_v2.gt_timing": module("timing",
                    install_gt_timing=lambda **kw: scope("timing", **kw)),
                "rl_agent.splitfusion_live_parity_runtime_v4.host_capture": module("host",
                    verify_prospective=lambda _: record),
            }
            patches.enter_context(mock.patch.dict(sys.modules, fake))
            patches.enter_context(mock.patch.object(optimization_scope, "install_cpu_gt_optimizations",
                lambda **kw: scope("optimization", optimized, **kw)))
            patches.enter_context(mock.patch.object(entries, "evidence_scope", lambda _: scope("evidence")))
            patches.enter_context(mock.patch.object(startup, "preserve_startup", lambda _: scope("startup")))
            writer = entries.write_x

            def write(path, value):
                events.append("report")
                writer(path, value)

            patches.enter_context(mock.patch.object(entries, "write_x", write))
            yield SimpleNamespace(attempt=attempt, output=output, record=record, config=config,
                events=events, optimized=optimized,
                report=lambda: json.loads((attempt / "GT_OPTIMIZATION_SCOPE.json").read_text()))

    def test_exact_outer_scope_order_bound_sources_and_finalized_report(self):
        with self.fixture() as f:
            self.assertEqual(entries.child(["fixture"]), 0)
            self.assertEqual(f.events, ["enter:optimization", "enter:ownership", "enter:timing", "enter:pose",
                "enter:evidence", "enter:startup", "retained", "exit:startup", "exit:evidence", "exit:pose",
                "exit:timing", "exit:ownership", "exit:optimization", "report"])
            report = f.report()
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["optimization_sources"], f.record["optimization_sources"])
            self.assertEqual(report["prospective_record_sha256"], C.sha(f.output / "PROSPECTIVE.json"))
            self.assertTrue(report["worker_drained"])
            self.assertTrue(report["ownership_manifest_complete"])
            self.assertTrue(report["optimization"]["retained_settings_unchanged"])

    def test_original_failure_preserved_and_all_scopes_exit_before_report(self):
        error = RuntimeError("retained error")
        with self.fixture(failure=error) as f:
            with self.assertRaises(RuntimeError) as caught:
                entries.child(["fixture"])
            self.assertIs(caught.exception, error)
            self.assertEqual(f.events[-2:], ["exit:optimization", "report"])
            self.assertEqual(f.report()["error"], {"type": "RuntimeError", "error": "retained error"})
            self.assertEqual(f.report()["status"], "FAILED")

    def test_prewarm_failure_never_enters_ownership(self):
        with self.fixture(prewarm_failure=True) as f:
            with self.assertRaisesRegex(RuntimeError, "prewarm failed"):
                entries.child(["fixture"])
            self.assertEqual(f.events, ["enter:optimization", "report"])
            self.assertFalse(f.report()["worker_drained"])
            self.assertIsNone(f.report()["optimization"])

    def test_actual_undrained_worker_fails_closed(self):
        with self.fixture(drained=False) as f:
            with self.assertRaisesRegex(RuntimeError, "did not close"):
                entries.child(["fixture"])
            self.assertEqual(f.report()["status"], "FAILED")
            self.assertFalse(f.report()["worker_drained"])
            self.assertFalse(f.report()["ownership_manifest_complete"])

    def test_missing_prospective_optimization_binding_cannot_install(self):
        with self.fixture() as f:
            del f.record["optimization_sources"]
            with self.assertRaisesRegex(ValueError, "prospectively bound"):
                entries.child(["fixture"])
            self.assertEqual(f.events, [])

    def test_stale_profiler_digest_cannot_install(self):
        with self.fixture() as f:
            f.config["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                entries.child(["fixture"])
            self.assertEqual(f.events, [])

    def test_report_create_only_and_evidence_error_cannot_replace_runtime_error(self):
        error = RuntimeError("original error")
        with self.fixture(failure=error) as f:
            path = f.attempt / "GT_OPTIMIZATION_SCOPE.json"
            path.write_text("preexisting evidence")
            with mock.patch.object(sys, "stderr"), self.assertRaises(RuntimeError) as caught:
                entries.child(["fixture"])
            self.assertIs(caught.exception, error)
            self.assertEqual(path.read_text(), "preexisting evidence")
        with self.fixture() as f:
            (f.attempt / "GT_OPTIMIZATION_SCOPE.json").write_text("preexisting evidence")
            with self.assertRaises(FileExistsError):
                entries.child(["fixture"])


class OptimizationOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import cv2
        cls.torch, cls.cv2 = torch, cv2
        cls.binding = optimization_scope.R.source_inventory()

    def test_real_scope_prewarms_restores_and_allows_only_cuda_lifecycle_transition(self):
        flag = [False]
        original_settings = optimization_scope.R.settings
        with mock.patch.object(self.torch.cuda, "is_initialized", side_effect=lambda: flag[0]):
            with optimization_scope.install_cpu_gt_optimizations(expected_sources=self.binding) as state:
                self.assertNotIn("cuda_initialized", state["settings_before"])
                self.assertEqual(set(state["settings_before"]), {"environment", "calling_thread_affinity",
                    "torch_threads", "torch_interop_threads", "opencv_threads"})
                flag[0] = True
            self.assertFalse(state["cuda_lifecycle_before"])
            self.assertTrue(state["cuda_lifecycle_after"])
            self.assertTrue(state["cuda_initialized_by_retained_path"])
            self.assertTrue(state["retained_settings_unchanged"])
            self.assertEqual(state["settings_before"], state["settings_after"])
            self.assertIsNone(state["source_integrity_error"])
            self.assertFalse(state["prewarm"]["source"]["state_or_history_created"])
            self.assertFalse(state["prewarm"]["target"]["frame_history_advanced"])
        self.assertIs(optimization_scope.R.settings, original_settings)

    def test_thread_cap_change_is_still_rejected(self):
        threads = [1]
        with mock.patch.object(self.torch, "get_num_threads", side_effect=lambda: threads[0]):
            with self.assertRaisesRegex(RuntimeError, "settings changed"):
                with optimization_scope.install_cpu_gt_optimizations(expected_sources=self.binding) as state:
                    threads[0] = 2
            self.assertFalse(state["retained_settings_unchanged"])

    def test_cuda_mask_change_is_still_rejected(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(RuntimeError, "settings changed"):
                with optimization_scope.install_cpu_gt_optimizations(expected_sources=self.binding) as state:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "changed-mask-observation-only"
            self.assertFalse(state["retained_settings_unchanged"])

    def test_stale_bound_inventory_cannot_prewarm(self):
        wrong = dict(self.binding, runtime="0" * 64)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            with optimization_scope.install_cpu_gt_optimizations(expected_sources=wrong):
                self.fail("stale binding admitted")

    def test_original_error_wins_over_setting_failure(self):
        error = RuntimeError("original retained failure")
        threads = [1]
        with mock.patch.object(self.torch, "get_num_threads", side_effect=lambda: threads[0]):
            with self.assertRaises(RuntimeError) as caught:
                with optimization_scope.install_cpu_gt_optimizations(expected_sources=self.binding) as state:
                    threads[0] = 2
                    raise error
            self.assertIs(caught.exception, error)
            self.assertFalse(state["retained_settings_unchanged"])


class StartupTests(unittest.TestCase):
    def test_stdlib_guard_calls_v4_entry_and_keeps_retained_records(self):
        writes = []
        with mock.patch.dict(startup.edge_entry.__globals__, {"write_x": lambda path, value: writes.append((path, value))}), \
                mock.patch.object(entries, "edge", return_value=0) as edge:
            self.assertEqual(startup.edge_entry(["--ready-file", "/work/torch_cache/ready.json"]), 0)
        edge.assert_called_once_with(["--ready-file", "/work/torch_cache/ready.json"])
        self.assertEqual([path.name for path, _ in writes], ["STARTED.json", "EXITED.json"])
        self.assertEqual(startup.edge_entry.__globals__["__package__"], entries.__package__)

    def test_ready_path_guard_precedes_entry(self):
        with mock.patch.object(entries, "edge", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(ValueError, "foreign"):
                startup.edge_entry(["--ready-file", "/tmp/foreign-ready.json"])


if __name__ == "__main__":
    unittest.main()

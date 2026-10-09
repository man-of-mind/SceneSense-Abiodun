"""Independent CPU-only preservation checks; never launch an application."""
import ast
import builtins
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from . import contract as C, entries, host_capture as host, partition_apply
from rl_agent.splitfusion_live_parity_runtime_v4 import contract as old_contract
from rl_agent.splitfusion_live_parity_runtime_v4 import entries as old_entries
from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as retained_host


class PreservationTests(unittest.TestCase):
    def test_original_scientific_metrics_and_new_one_shot_identity(self):
        self.assertEqual(C.metrics(), old_contract.metrics())
        self.assertEqual(C.BUDGET, 100)
        self.assertEqual(C.IMAGE_ID, old_contract.IMAGE_ID)
        for name in ("ID", "TOKEN", "CHILD", "EDGE", "RUNNER", "PARTITION"):
            self.assertNotEqual(getattr(C, name), getattr(old_contract, name), name)
        self.assertTrue(C.CHILD.startswith("rl_agent.splitfusion_single_gpu_profiler_v1.child_v1"))
        self.assertTrue(C.EDGE.startswith("rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1"))
        metrics = C.metrics()
        self.assertEqual(metrics["actor_state_dim"], 22)
        self.assertEqual(metrics["inclusive_deadline_ms"], 170)
        self.assertTrue(metrics["quality_evaluation_inside_deadline"])
        self.assertTrue(metrics["previous_quality_in_state"])
        self.assertFalse(metrics["training_started"])

    def test_host_launch_is_exact_retained_source_with_only_two_command_substitutions(self):
        expected = inspect.getsource(retained_host.launch)
        for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                              ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)):
            self.assertEqual(expected.count(before), 1)
            expected = expected.replace(before, after)
        seen = []
        def capture_compile(source, filename, mode):
            seen.append(source)
            return builtins.compile(source, filename, mode)
        def intercept_execute(code, namespace):
            self.assertIs(namespace["_verify_prospective"], host.verify_prospective)
            self.assertIs(namespace["_verify_prerequisite"], host.verify_prerequisite)
            namespace["launch"] = lambda output, execute_token: (output, execute_token)
        with mock.patch.object(host, "compile", capture_compile, create=True), \
                mock.patch.object(host, "exec", intercept_execute, create=True):
            result = host.launch("unlaunched", execute_token="diagnostic-only")
        self.assertEqual(result, ("unlaunched", "diagnostic-only"))
        self.assertEqual(seen, [expected])

    def _guarded_launch(self, directory, environment, *, marker=None):
        root = Path(directory)
        (root / "registration.json").write_text(json.dumps({"arms": {"C2": {"execute_token": C.TOKEN}}}))
        if marker:
            (root / marker).touch()
        record = {"worktree": str(root), "retained_code": str(root)}
        with mock.patch.object(host, "verify_prospective", return_value=record), \
                mock.patch.object(host, "verify_prerequisite", return_value={}), \
                mock.patch.dict(retained_host.os.environ, environment, clear=True), \
                mock.patch.object(retained_host.subprocess, "run", side_effect=AssertionError("unexpected launch")), \
                mock.patch.object(retained_host.subprocess, "Popen", side_effect=AssertionError("unexpected process")):
            return host.launch(root, execute_token=C.TOKEN)

    def test_live_parent_rejects_caps_injection_and_offline_empty_cuda_mask(self):
        for environment in ({"OMP_NUM_THREADS": "1"}, {"MKL_NUM_THREADS": "1"},
                            {"LD_PRELOAD": "forbidden.so"}, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.subTest(environment=environment), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self._guarded_launch(directory, environment)

    def test_existing_attempt_cannot_be_retried_before_any_subprocess(self):
        for marker in ("controller_events.jsonl", "capture"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileExistsError):
                    self._guarded_launch(directory, {}, marker=marker)

    def test_controller_lineage_uses_new_complete_source_inventory(self):
        source = {"new_runtime.py": "a" * 64, "optimized_gt.py": "b" * 64}
        optimized = {"runtime.py": "d" * 64}
        with mock.patch.object(C, "source_inventory", return_value=source) as inventory, \
                mock.patch.object(C, "optimization_inventory", return_value=optimized) as optimization:
            result = C.lineage("diagnostic-root", "c" * 64)
        inventory.assert_called_once_with("diagnostic-root")
        optimization.assert_called_once_with("diagnostic-root")
        expected = C.canonical(dict(actor_boundary_sha256="c" * 64, sources=source,
            optimization_sources=optimized,
            metrics=C.metrics(), controller="RewardHoldControllerV2", engine="Run4DecisionEngineV2",
            state="ORIGINAL_RUN5_22D_QUALITY_AWARE", fallback="mode11_q9800_a71"))
        self.assertEqual(result, expected)

    def test_partition_uses_new_edge_matcher_without_mutating_retained_globals(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as original
        namespace = {"EDGE_MODULE": "retained-edge"}
        exec("def apply(output, retained_code):\n    return EDGE_MODULE, output, retained_code\n", namespace)
        function = namespace["apply"]
        with mock.patch.object(original, "apply", function):
            self.assertEqual(partition_apply.apply("output", "retained"), (C.EDGE, "output", "retained"))
        self.assertEqual(function.__globals__["EDGE_MODULE"], "retained-edge")

    def test_edge_dispatch_has_no_new_gt_imports_or_execution(self):
        current = ast.parse(inspect.getsource(entries.run_edge_only))
        previous = ast.parse(inspect.getsource(old_entries.run_edge_only))
        self.assertEqual(ast.dump(current, include_attributes=False), ast.dump(previous, include_attributes=False))
        text = inspect.getsource(entries.edge)
        self.assertNotIn("splitfusion_live_gt_optimization", text)
        self.assertNotIn("composition", text)

    def test_optimization_binding_is_exact_replayed_bytes_not_freshly_accepted_code(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(C.optimization_inventory(root), C.OPTIMIZATION_SOURCES)
        with tempfile.TemporaryDirectory() as scratch:
            candidate_root = Path(scratch)
            target = candidate_root / C.OPTIMIZATION_PACKAGE
            target.mkdir(parents=True)
            for name in C.OPTIMIZATION_SOURCES:
                shutil.copyfile(root / C.OPTIMIZATION_PACKAGE / name, target / name)
            self.assertEqual(C.optimization_inventory(candidate_root), C.OPTIMIZATION_SOURCES)
            with (target / "candidate.py").open("ab") as stream:
                stream.write(b"\n# unreviewed source mutation\n")
            with self.assertRaisesRegex(ValueError, "exactly replayed optimization"):
                C.optimization_inventory(candidate_root)


if __name__ == "__main__":
    unittest.main()

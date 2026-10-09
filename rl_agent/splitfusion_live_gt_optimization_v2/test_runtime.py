"""Independent CPU-only checks of the additive v2 live-scope contract."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import cv2  # noqa: F401 - make process-settings keys stable before each scope
import torch

from rl_agent.splitfusion_live_gt_optimization_v1 import runtime as R1
from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_gt_optimization_v1 import target_builder as V1T
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_parity_runtime_v4 import contract as C4
from rl_agent.splitfusion_live_parity_runtime_v4 import optimization_scope as V4
from rl_agent.splitfusion_live_parity_runtime_v5 import contract as C5
from rl_agent.splitfusion_live_parity_runtime_v5 import optimization_scope as V5
from . import candidate, runtime as R


def bindings():
    return (S.LiveCompleteTargetProducer, A.build_complete_v010, E.prepare_target,
            E.restore_prepared_target, T.write_bundle, G.write_source_archive)


class AdditiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.original = bindings()
        self.settings = R.configuration_settings()
        self.cuda_initialized = torch.cuda.is_initialized()
        self.inventory = R.source_inventory()

    def tearDown(self):
        self.assertEqual(bindings(), self.original)
        self.assertEqual(R.configuration_settings(), self.settings)
        self.assertEqual(torch.cuda.is_initialized(), self.cuda_initialized)
        self.assertFalse(R1._LOCK.locked())
        self.assertFalse(candidate._LOCK.locked())

    def test_exact_nested_install_and_full_restoration(self):
        with R.install_cpu_gt_optimizations(expected_sources=self.inventory) as state:
            active = bindings()
            self.assertTrue(issubclass(active[0], V1S.OptimizedSourceProducer))
            self.assertIsNot(active[0], V1S.OptimizedSourceProducer)
            self.assertIs(active[1].__globals__, vars(V1T))
            self.assertEqual(active[1].__qualname__, "install_target_builder.<locals>.installed")
            for current, original in zip(active, self.original):
                self.assertIsNot(current, original)
            self.assertEqual(state["sources"], self.inventory)
            self.assertEqual(state["retained_v1_sources"], C4.OPTIMIZATION_SOURCES)
            self.assertEqual(state["schema"], "splitfusion.cpu_gt_optimization.scope.v2")
            self.assertEqual(state["new_python_threads"], 0)
            self.assertEqual(state["new_queues"], 0)
            self.assertFalse(state["new_gpu_work"])
            self.assertFalse(state["forensic_writes_deferred"])
            self.assertTrue(state["candidate"]["fast_labels"])
            self.assertTrue(state["candidate"]["fast_metadata"])
            self.assertTrue(state["candidate"]["fsync_unchanged"])
            self.assertEqual(R.configuration_settings(), self.settings)
        self.assertTrue(state["retained_settings_unchanged"])
        self.assertIsNone(state["source_integrity_error"])
        self.assertEqual(state["settings_before"], state["settings_after"])
        self.assertFalse(state["cuda_initialized_by_retained_path"])

    def test_inventory_includes_runtime_and_matches_prospective_v5(self):
        self.assertIn("runtime.py", self.inventory)
        self.assertIn("candidate.py", self.inventory)
        self.assertNotIn("test_runtime.py", self.inventory)
        self.assertEqual(self.inventory, C5.OPTIMIZATION_SOURCES)
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(C5.optimization_inventory(root), self.inventory)
        self.assertIs(V5.install_cpu_gt_optimizations, R.install_cpu_gt_optimizations)
        self.assertIs(V5.configuration_settings, V4.configuration_settings)
        self.assertEqual(C5.BUDGET, 100)
        self.assertEqual(C5.IMAGE_ID, C4.IMAGE_ID)

    def test_foreign_or_missing_binding_rejected_before_prewarm(self):
        foreign = dict(self.inventory, **{"runtime.py": "0" * 64})
        for supplied in (None, {}, [], foreign):
            with self.subTest(binding=supplied), mock.patch.object(V1S, "prewarm") as prewarm:
                with self.assertRaisesRegex(ValueError, "source inventory"):
                    with R.install_cpu_gt_optimizations(expected_sources=supplied):
                        self.fail("foreign binding admitted")
                prewarm.assert_not_called()

    def test_retained_v1_source_drift_also_rejected_before_install(self):
        foreign = dict(R1.source_inventory(), **{"runtime.py": "0" * 64})
        with mock.patch.object(R1, "source_inventory", return_value=foreign):
            with self.assertRaisesRegex(ValueError, "source inventory"):
                with R.install_cpu_gt_optimizations(expected_sources=self.inventory):
                    self.fail("retained source drift admitted")

    def test_source_drift_after_yield_fails_closed_after_restoration(self):
        current = dict(self.inventory)
        with mock.patch.object(R, "source_inventory", side_effect=lambda: dict(current)):
            with self.assertRaisesRegex(RuntimeError, "source inventory"):
                with R.install_cpu_gt_optimizations(expected_sources=self.inventory) as state:
                    current["candidate.py"] = "0" * 64
        self.assertIn("source inventory", state["source_integrity_error"])
        self.assertTrue(state["retained_settings_unchanged"])

    def test_caller_exception_preserved_when_both_guards_detect_changes(self):
        error = LookupError("original retained-path exception")
        current = dict(self.inventory)
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}), \
             mock.patch.object(R, "source_inventory", side_effect=lambda: dict(current)):
            with self.assertRaises(LookupError) as caught:
                with R.install_cpu_gt_optimizations(expected_sources=self.inventory) as state:
                    current["candidate.py"] = "0" * 64
                    os.environ["OMP_NUM_THREADS"] = "2"
                    raise error
            self.assertIs(caught.exception, error)
        self.assertFalse(state["retained_settings_unchanged"])
        self.assertIn("source inventory", state["source_integrity_error"])

    def test_retained_settings_change_rejected_on_normal_exit(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "retained process settings changed"):
                with R.install_cpu_gt_optimizations(expected_sources=self.inventory) as state:
                    os.environ["OMP_NUM_THREADS"] = "2"
        self.assertFalse(state["retained_settings_unchanged"])
        self.assertIsNone(state["source_integrity_error"])

    def test_cuda_lifecycle_is_observed_without_weakening_settings_guard(self):
        # CPU-only simulation: no CUDA API or initialization occurs.
        lifecycle = [False]
        with mock.patch.object(torch.cuda, "is_initialized", side_effect=lambda: lifecycle[0]):
            with R.install_cpu_gt_optimizations(expected_sources=self.inventory) as state:
                lifecycle[0] = True
        self.assertFalse(state["cuda_lifecycle_before"])
        self.assertTrue(state["cuda_lifecycle_after"])
        self.assertTrue(state["cuda_initialized_by_retained_path"])
        self.assertTrue(state["retained_settings_unchanged"])
        self.assertNotIn("cuda_initialized", state["settings_before"])
        self.assertNotIn("cuda_initialized", state["settings_after"])

    def test_nested_and_concurrent_scope_rejected_without_displacing_active_scope(self):
        with R.install_cpu_gt_optimizations(expected_sources=self.inventory):
            active = bindings()
            with self.assertRaisesRegex(RuntimeError, "nested|concurrent"):
                with R.install_cpu_gt_optimizations(expected_sources=self.inventory):
                    self.fail("nested scope admitted")
            self.assertEqual(bindings(), active)
            outcomes = []

            def contender():
                try:
                    with R.install_cpu_gt_optimizations(expected_sources=self.inventory):
                        outcomes.append("admitted")
                except BaseException as error:
                    outcomes.append(error)

            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], RuntimeError)
            self.assertRegex(str(outcomes[0]), "nested|concurrent")
            self.assertEqual(bindings(), active)

    def test_caller_binding_mutation_does_not_change_owned_identity(self):
        supplied = dict(self.inventory)
        with R.install_cpu_gt_optimizations(expected_sources=supplied) as state:
            supplied.clear()
            self.assertEqual(state["sources"], self.inventory)
        self.assertIsNone(state["source_integrity_error"])

    def test_bad_candidate_entry_restores_retained_v1_and_releases_locks(self):
        marker = lambda *args, **kwargs: None
        with mock.patch.object(E, "prepare_target", marker):
            foreign = bindings()
            with self.assertRaisesRegex(RuntimeError, "original GT entrypoints"):
                with R.install_cpu_gt_optimizations(expected_sources=self.inventory):
                    self.fail("foreign candidate entrypoint admitted")
            self.assertEqual(bindings(), foreign)
        with R.install_cpu_gt_optimizations(expected_sources=self.inventory):
            self.assertIsNot(E.prepare_target, self.original[2])

    def test_symlink_sources_refused_without_reading_foreign_python(self):
        with tempfile.TemporaryDirectory(prefix="gt-v2-runtime-source-") as directory:
            folder = Path(directory)
            (folder / "runtime.py").symlink_to(Path(__file__))
            with mock.patch.object(R, "PACKAGE", folder):
                with self.assertRaisesRegex(ValueError, "regular v2 optimization sources"):
                    R.source_inventory()


if __name__ == "__main__":
    unittest.main()

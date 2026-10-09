import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2 import gt_timing
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import runtime, source_geometry, target_builder


class IntegrationTests(unittest.TestCase):
    def test_only_two_gt_bindings_changed_and_restored(self):
        old = (S.LiveCompleteTargetProducer, A.build_complete_v010, E.prepare_target, T.write_bundle)
        before = runtime.settings()
        with runtime.install_cpu_gt_optimizations(expected_sources=runtime.source_inventory()) as state:
            self.assertIs(S.LiveCompleteTargetProducer, source_geometry.OptimizedSourceProducer)
            self.assertIsNot(A.build_complete_v010, old[1])
            self.assertIs(E.prepare_target, old[2])
            self.assertIs(T.write_bundle, old[3])
        self.assertEqual(old, (S.LiveCompleteTargetProducer, A.build_complete_v010, E.prepare_target, T.write_bundle))
        self.assertEqual(before, runtime.settings())
        self.assertTrue(state["retained_settings_unchanged"])
        self.assertIsNone(state["source_integrity_error"])

    def test_foreign_binding_cannot_install(self):
        old = S.LiveCompleteTargetProducer
        wrong = runtime.source_inventory()
        wrong["runtime.py"] = "0"*64
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            with runtime.install_cpu_gt_optimizations(expected_sources=wrong):
                self.fail("foreign code admitted")
        self.assertIs(S.LiveCompleteTargetProducer, old)

    def test_owned_binding_and_exception_restore(self):
        supplied = runtime.source_inventory()
        expected = dict(supplied)
        old = (S.LiveCompleteTargetProducer, A.build_complete_v010)
        error = RuntimeError("original caller failure")
        with self.assertRaises(RuntimeError) as caught:
            with runtime.install_cpu_gt_optimizations(expected_sources=supplied) as state:
                supplied.clear()
                state["sources"].clear()
                raise error
        self.assertIs(caught.exception, error)
        self.assertIsNone(state["source_integrity_error"])
        self.assertEqual(runtime.source_inventory(), expected)
        self.assertEqual(old, (S.LiveCompleteTargetProducer, A.build_complete_v010))

    def test_nested_install_is_refused_and_parent_survives(self):
        binding = runtime.source_inventory()
        with runtime.install_cpu_gt_optimizations(expected_sources=binding):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with runtime.install_cpu_gt_optimizations(expected_sources=binding):
                    self.fail("nested scope admitted")
            self.assertIs(S.LiveCompleteTargetProducer, source_geometry.OptimizedSourceProducer)

    def test_source_failure_on_exception_is_recorded_not_substituted(self):
        original_verify = runtime.verify_sources
        calls = []
        def check(expected):
            calls.append(True)
            if len(calls) > 1:
                raise ValueError("controlled changed source")
            return original_verify(expected)
        with mock.patch.object(runtime, "verify_sources", check):
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                with runtime.install_cpu_gt_optimizations(expected_sources=runtime.source_inventory()) as state:
                    raise RuntimeError("original failure")
        self.assertEqual(state["source_integrity_error"], "controlled changed source")

    def test_retained_timing_wraps_candidate_and_restores_it(self):
        # Exercise actual instrumentation binding order, not a duplicate timer.
        with tempfile.TemporaryDirectory() as directory:
            with runtime.install_cpu_gt_optimizations(expected_sources=runtime.source_inventory()):
                candidate = S.LiveCompleteTargetProducer.build
                with gt_timing.install_gt_timing(attempt_dir=Path(directory), max_frames=1,
                                                drain_check=lambda: True) as log:
                    self.assertIs(S.LiveCompleteTargetProducer.build.__wrapped__, candidate)
                    self.assertIs(E.prepare_target.__wrapped__, log.wrap("unused", E.prepare_target.__wrapped__).__wrapped__)
                self.assertIs(S.LiveCompleteTargetProducer.build, candidate)
            self.assertTrue((Path(directory)/"complete_gt_stage_timing/MANIFEST.json").is_file())


if __name__ == "__main__":
    unittest.main()

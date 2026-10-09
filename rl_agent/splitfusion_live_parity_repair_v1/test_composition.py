"""Retained child/edge entry composition with no external process or CUDA."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import rl_agent

from . import composition as C, gt_bridge, prediction_archive

PROFILER = Path(__file__).resolve().parents[2].parent / "abiodun_run4_single_gpu_profiler_v1"
if importlib.util.find_spec("rl_agent.splitfusion_single_gpu_profiler_v1") is None:
    rl_agent.__path__.append(str(PROFILER / "rl_agent"))  # explicit tests only


class CompositionTests(unittest.TestCase):
    def test_frozen_teardown_fixture_with_image_lookup_mocked(self):
        # Frozen test mocks all launchers but predates the added Docker-image
        # admission lookup. Bind that missing test dependency explicitly;
        # never inspect/start a real container in this CPU regression.
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import test_phase6_live_integration_v2 as OLD
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL
        with mock.patch.object(EL, "resolve_admitted_image", return_value={"source": "CPU_TEST_ONLY"}):
            case = OLD.LegacyAndLifecycleTest("test_teardown_is_attempted_for_every_resource_on_child_failure")
            case.test_teardown_is_attempted_for_every_resource_on_child_failure()

    def test_real_retained_child_installer_sees_new_scope_and_restores(self):
        gt_bridge.bind_legacy_namespace()
        from rl_agent.splitfusion_run5_live_avo_v1 import child_entry as CHILD, live_gt as LG
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as LP
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
        from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        from . import perception
        old = (LG._INSTALLED, Q.QualitySceneSnapshotSource, Q.QualityPassiveSplitCollector,
               LP._prepare_live_input, PW.RADAR_SHAPE, CHILD.EDGE_MODULE)
        with tempfile.TemporaryDirectory() as root:
            def delegate(argv):
                self.assertIs(LP._prepare_live_input, perception.prepare_live_input)
                self.assertEqual(PW.RADAR_SHAPE, (4,432,768))
                result = LG.install_ue_gt_seams(attempt_dir=Path(root))
                self.assertEqual(result["new_gt_helper_threads"], 0)
                self.assertEqual(Q.QualitySceneSnapshotSource.__name__, "CompleteSource")
                self.assertEqual(Q.QualityPassiveSplitCollector.__name__, "CompleteCollector")
                self.assertTrue(LG._INSTALLED)
                return 17
            with mock.patch.object(CHILD, "main", side_effect=delegate):
                result = C.run_ue_child(["--attempt-dir", root, "--transmitted-budget", "300"], max_frames=300,
                    edge_module="rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_parity_repair")
                self.assertEqual(result, 17)
        self.assertEqual((LG._INSTALLED,Q.QualitySceneSnapshotSource,Q.QualityPassiveSplitCollector,
                          LP._prepare_live_input,PW.RADAR_SHAPE,CHILD.EDGE_MODULE), old)

    def test_legacy_edge_and_missing_attempt_refused(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import child_entry as CHILD
        with self.assertRaises(ValueError):
            C.run_ue_child([], max_frames=1, edge_module=CHILD.EDGE_MODULE)
        with self.assertRaisesRegex(ValueError, "legacy"):
            C.run_ue_child(["--attempt-dir", "/tmp/not-created"], max_frames=1, edge_module=CHILD.EDGE_MODULE)

    def test_prediction_archive_copied_before_scratch_teardown_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root/"scratch"/"profiler"/"prediction_archive"
            prediction_archive.PredictionArchive(max_frames=1).flush(source)
            copied = C.preserve_prediction_archive(root/"scratch", root/"retained")
            self.assertTrue(copied["preserved_before_teardown"])
            self.assertEqual((source/"MANIFEST.json").read_bytes(), (root/"retained"/"MANIFEST.json").read_bytes())
            with self.assertRaises(FileExistsError):
                C.preserve_prediction_archive(root/"scratch", root/"retained")
            with self.assertRaises(ValueError):
                C.preserve_prediction_archive(root/"missing", root/"unused")

    def test_edge_evaluator_inside_retained_partition_profiler_entry(self):
        from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        gate = mock.Mock()
        before = E.Run4EvaluatorV2._evaluate_with
        def delegate(argv):
            gate.install.assert_called_once()
            self.assertIsNot(E.Run4EvaluatorV2._evaluate_with, before)
            self.assertIn("PROSPECTIVE_LIVE_CONSERVATIVE", W.LIVE_MEASUREMENT_SEMANTICS)
            return 0
        old_path = list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"edge_gate": gate}), \
                 mock.patch.object(ORDERED, "main", side_effect=delegate):
                self.assertEqual(C.run_edge(["--run4-config", str(Path(root)/"edge.json")],max_frames=300), 0)
        finally:
            sys.path[:] = old_path
        self.assertIs(E.Run4EvaluatorV2._evaluate_with, before)


if __name__ == "__main__":
    unittest.main()

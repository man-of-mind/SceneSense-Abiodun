"""Independent CPU checks of the deployed scope nesting and preservation path."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import rl_agent

from . import contract as C, entries, host_capture, pose


ROOT = Path(__file__).resolve().parents[2]
PROFILER = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1"
if importlib.util.find_spec("rl_agent.splitfusion_single_gpu_profiler_v1") is None:
    # Explicit CPU-test infrastructure only; staged execution resolves locally.
    rl_agent.__path__.append(str(PROFILER / "rl_agent"))


class IntegrationReviewTests(unittest.TestCase):
    def test_actual_collector_factory_keeps_owned_pose_below_complete_gt(self):
        from rl_agent.splitfusion_live_parity_repair_v1 import composition, gt_bridge
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        gt_bridge.bind_legacy_namespace()
        old = (LG._INSTALLED, LG.build_live_gt_collector, Q.QualityPassiveSplitCollector,
               Q.QualitySceneSnapshotSource, P.actor_world_matrix, P.actor_world_inverse_matrix)
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary) / "attempt"
            attempt.mkdir()
            with pose.install_ue_pose(attempt_dir=attempt, max_frames=100), \
                    entries.evidence_scope(attempt), \
                    composition.ue_scope(attempt_dir=attempt, max_frames=100):
                LG.install_ue_gt_seams(attempt_dir=attempt)
                names = [cls.__name__ for cls in Q.QualityPassiveSplitCollector.__mro__]
                self.assertEqual(names[:3], ["CompleteCollector", "OwnedPoseCollector", "LiveAvoCollector"])
                self.assertEqual(names.count("LiveAvoCollector"), 1)
                self.assertEqual(Q.QualitySceneSnapshotSource.__name__, "CompleteSource")
            self.assertTrue((attempt / "pose_archive" / "MANIFEST.json").is_file())
        self.assertEqual((LG._INSTALLED, LG.build_live_gt_collector, Q.QualityPassiveSplitCollector,
                          Q.QualitySceneSnapshotSource, P.actor_world_matrix, P.actor_world_inverse_matrix), old)

    def test_complete_target_redirect_preserves_old_exact_leaf_inventory(self):
        from rl_agent.splitfusion_live_parity_repair_v1 import composition
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            with entries.evidence_scope(attempt), \
                    composition.ue_scope(attempt_dir=attempt, max_frames=100):
                result = LG.preserve_avo_evidence([], attempt / "run5_live_avo_inputs")
            old_leaf = attempt / "run5_live_avo_inputs"
            manifest = json.loads((old_leaf / "MANIFEST.json").read_bytes())
            self.assertEqual({p.name for p in old_leaf.iterdir()}, set(manifest["sha256"]) | {"MANIFEST.json"})
            self.assertFalse((old_leaf / "complete_targets").exists())
            self.assertTrue((attempt / "parity_complete_targets" / "MANIFEST.json").is_file())
            self.assertEqual(result["complete_targets"]["directory"], str(attempt / "parity_complete_targets"))

    def test_real_async_close_survives_profiler_subclass_and_both_archives_copy(self):
        from rl_agent.splitfusion_live_parity_repair_v1 import composition, runtime
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import Pipeline, processor
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
        from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
        from rl_agent.splitfusion_single_gpu_profiler_v1 import recorder_v1 as RV, seams_ue_v1 as SU
        # Compile the actual retained evaluator subclass, avoiding unrelated
        # global profiler mutations and its GPU-aware inference wrappers.
        syntax = ast.parse(inspect.getsource(EP.install_edge_seams))
        node = next(n for n in syntax.body[0].body
                    if isinstance(n, ast.ClassDef) and n.name == "ProfiledEvaluatorV1")
        globals_ = dict(EP.__dict__, E=E, RV=RV, rec=mock.MagicMock())
        base = E.Run4EvaluatorV2
        prior = (base.close, base.submit, base._evaluate_with, runtime.install_edge_evaluation)
        prepared = Pipeline().step()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scratch, attempt = root / "scratch", root / "attempt"
            profile = scratch / "profiler"
            profile.mkdir(parents=True)
            attempt.mkdir()
            try:
                with pose.install_edge_pose(archive_directory=profile / "pose_archive", max_frames=100), \
                        entries.edge_close_evidence(profile), \
                        runtime.install_edge_evaluation(archive_directory=profile / "prediction_archive", max_frames=100):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "<retained-evaluator-subclass>", "exec"), globals_)
                    E.Run4EvaluatorV2 = globals_["ProfiledEvaluatorV1"]
                    PW.publish_ready_after_warmup(
                        lambda: {"completed": True, "modes_warmed": list(range(12))}, lambda: None,
                        report_path=profile / "cpu_test_prewarm.json")
                    ticket = processor().process(prepared.wire, edge_timing={}).evaluation
                    self.assertIsNotNone(ticket)
                    worker = E.Run4EvaluatorV2(spec=W.load_run4_quality_spec(ROOT), send=lambda data: None,
                        read_ground_truth=lambda **kw: G.read_ground_truth(root / "absent_gt", **kw),
                        match_distance_m=3., gt_timeout_s=.005, probe_timeout_s=.001, poll_interval_s=.001)
                    worker.start()
                    worker.submit(ticket)
                    result = worker.close()
                    self.assertFalse(result["worker_alive"])
                    self.assertTrue(result["parity_archive_complete"])
                    self.assertTrue(result["pose_archive_complete"])
                    self.assertEqual(result["gt_expired"], 1)
                    self.assertEqual(result["parity_archive"]["frames"], 1)
                    self.assertEqual(result["pose_archive"]["frames"], 1)
                    self.assertEqual(json.loads((profile / "evaluator_close_result.json").read_text()), result)
            finally:
                E.Run4EvaluatorV2 = base
            self.assertEqual((base.close, base.submit, base._evaluate_with, runtime.install_edge_evaluation), prior)
            with entries.evidence_scope(attempt), \
                    composition.ue_scope(attempt_dir=attempt, max_frames=100):
                destination = attempt / "profiler" / "edge"
                copied = SU.copy_edge_profile(str(scratch), destination)
            self.assertTrue(copied["pose_archive_preserved"])
            self.assertTrue(copied["parity_predictions_preserved"])
            for path in profile.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (destination / path.relative_to(profile)).read_bytes())

    def test_unprofiled_child_fails_before_any_context_or_delegate(self):
        from rl_agent.splitfusion_live_parity_repair_v1 import composition
        from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
        with mock.patch.object(BASE, "load_config", return_value=None), \
                mock.patch.object(pose, "install_ue_pose") as install, \
                mock.patch.object(composition, "run_ue_child") as delegate:
            with self.assertRaisesRegex(ValueError, "profiler configuration"):
                entries.child(["--attempt-dir", "/tmp/not_created_parity_review"])
            install.assert_not_called()
            delegate.assert_not_called()

    def test_retained_launch_changes_exactly_two_module_literals(self):
        original = inspect.getsource(host_capture.H.launch)
        expected = original.replace("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER).replace(
            "rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)
        captured = {}
        sentinel = object()
        def compile_source(source, filename, mode):
            captured["source"] = source
            self.assertEqual(mode, "exec")
            return sentinel
        def execute(code, namespace):
            self.assertIs(code, sentinel)
            self.assertIs(namespace["_verify_prospective"], host_capture.verify_prospective)
            self.assertIs(namespace["_verify_prerequisite"], host_capture.verify_prerequisite)
            namespace["launch"] = lambda output, execute_token: (output, execute_token)
        with mock.patch.object(host_capture, "compile", side_effect=compile_source, create=True), \
                mock.patch.object(host_capture, "exec", side_effect=execute, create=True):
            self.assertEqual(host_capture.launch("owned-attempt", execute_token="not-executed"),
                             ("owned-attempt", "not-executed"))
        self.assertEqual(captured["source"], expected)

    def test_new_avo_verifier_is_scoped_to_retained_postflight(self):
        from . import postflight as repaired
        from rl_agent.splitfusion_run5_live_avo_v1 import verify_live_gt_capture as V
        old_verify, old_prospective = V.verify_capture, host_capture.H._verify_prospective
        def retained(output):
            self.assertIs(V.verify_capture, repaired.verify_avo_capture)
            self.assertIsNot(host_capture.H._verify_prospective, old_prospective)
            return {"passed": False}
        def final(output):
            self.assertIs(V.verify_capture, old_verify)
            self.assertIs(host_capture.H._verify_prospective, old_prospective)
            return {"passed": True, "scope": "new exact complete GT"}
        with mock.patch.object(host_capture, "verify_prospective", return_value={}), \
                mock.patch.object(host_capture.H, "postflight", side_effect=retained), \
                mock.patch.object(repaired, "postflight", side_effect=final):
            self.assertEqual(host_capture.postflight("owned-attempt")["scope"], "new exact complete GT")
        self.assertIs(V.verify_capture, old_verify)
        self.assertIs(host_capture.H._verify_prospective, old_prospective)


if __name__ == "__main__":
    unittest.main()

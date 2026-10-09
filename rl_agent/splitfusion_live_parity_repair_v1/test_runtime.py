"""Real Phase6 processor/evaluator composition; no GPU, radio or CARLA."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from . import evaluation, perception, runtime
from . import test_evaluation as TE


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TE.CompleteEvaluationTests.setUpClass()
        fixture = TE.CompleteEvaluationTests()
        cls.args = fixture.inputs_for()

    def make_ticket(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import Pipeline, processor
        pipeline = Pipeline()
        pipeline.frame = self.args["prepared"].frame_id - 1
        columns = TE.CompleteEvaluationTests.original.PREDICTION_FIELDS
        records = [dict(class_name=r["class_name"], frame_id=self.args["prepared"].frame_id,
                        **{k: float(r[k]) for k in columns})
                   for r in self.args["predictions"]]
        labels = np.zeros((432, 768), np.uint8)
        produced = processor(compute=lambda **kw: (records, labels)).process(pipeline.step().wire, edge_timing={})
        return produced.evaluation

    def test_ue_geometry_installs_before_sensor_start_and_unwinds(self):
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
        klass = P.PassiveSplitCollector
        original = B._prepare_live_input
        host = SimpleNamespace(sensors=[], model_size=(768, 448),
            intrinsics=perception.radar_projection_intrinsics().copy())
        host.intrinsics[1, 2] = 224
        observed = []
        def spawn(obj):
            observed.append((obj.model_size, obj.intrinsics[1, 2]))
            obj.sensors.append("native camera unchanged")
        with mock.patch.object(klass, "_spawn_sensors", spawn):
            with runtime.install_ue_perception():
                self.assertIs(B._prepare_live_input, perception.prepare_live_input)
                klass._spawn_sensors(host)
            self.assertIs(klass._spawn_sensors, spawn)
        self.assertEqual(observed, [((768, 432), 216.)])
        self.assertIs(B._prepare_live_input, original)

    def test_real_evaluator_exact_reward_once_and_archive_after_send(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
        from . import target_transport as T
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt
        value = self.make_ticket()
        expected = evaluation.evaluate_prepared(identity=MP.run4_identity(value.envelope),
            prepared=self.args["prepared"], predicted_labels=value.predicted_mask,
            predictions=value.records, avo=self.args["avo"])
        spec = W.load_run4_quality_spec(Path(__file__).resolve().parents[2])
        expected_feedback, expected_reason = W.quality_feedback(spec, expected.measurement, value.envelope)
        sent = []
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "predictions"
            bundle = SimpleNamespace(prepared=self.args["prepared"], manifest_sha256="a"*64,
                                     manifest={"ready_wall_ns": time.time_ns()})
            avo = {"avo": self.args["avo"], "avo_sidecar_sha256": "b"*64,
                   "avo_ready_wall_ns": time.time_ns()}
            with mock.patch.object(T, "read_bundle", return_value=bundle), \
                 mock.patch.object(live_gt, "read_avo_bundle", return_value=avo), \
                 runtime.install_edge_evaluation(archive_directory=destination, max_frames=2) as state:
                from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
                def send(payload):
                    self.assertFalse(destination.exists())
                    sent.append(payload)
                worker = E.Run4EvaluatorV2(spec=spec, send=send,
                    read_ground_truth=lambda **kw: G.read_ground_truth(Path(root), **kw),
                    match_distance_m=3., gt_timeout_s=.2)
                worker.start()
                worker.submit(value)
                result = worker.close()
                self.assertFalse(result["worker_alive"])
                self.assertTrue(result["parity_archive_complete"])
                self.assertEqual(state["archive_errors"], [])
            self.assertEqual(sent, [W.encode_feedback(expected_feedback, expected_reason)])
            self.assertEqual(result["parity_archive"]["frames"], 1)
            self.assertEqual(result["parity_archive"]["evaluated_frames"], 1)

    def test_real_ue_prewarm_accepts_repaired_content_and_restores(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import CONTRACT, runtimes
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
        original_shape = PW.RADAR_SHAPE
        ue, _, _ = runtimes()
        with runtime.install_ue_perception():
            result = PW.warm_ue(ue, CONTRACT,
                prepare_input=lambda f, r: B._prepare_live_input(f, r, "cpu"), sync=lambda: None)
            self.assertTrue(result["completed"])
            self.assertEqual(result["modes_warmed"], list(range(12)))
            self.assertEqual(result["input_shape"], [1, 7, 448, 768])
        self.assertEqual(PW.RADAR_SHAPE, original_shape)

    def test_missing_complete_target_never_uses_legacy_raw_gt(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
        value = self.make_ticket()
        sent = []
        spec = W.load_run4_quality_spec(Path(__file__).resolve().parents[2])
        old_read = G.read_ground_truth
        with tempfile.TemporaryDirectory() as root:
            with runtime.install_edge_evaluation(archive_directory=Path(root)/"predictions", max_frames=1):
                worker = E.Run4EvaluatorV2(spec=spec, send=sent.append,
                    read_ground_truth=lambda **kw: G.read_ground_truth(Path(root), **kw),
                    match_distance_m=3., gt_timeout_s=.01, probe_timeout_s=.001)
                worker.start()
                worker.submit(value)
                result = worker.close()
            self.assertIs(G.read_ground_truth, old_read)
            self.assertEqual(result["gt_expired"], 1)
            self.assertEqual(result["parity_archive"]["frames"], 1)
            self.assertEqual(result["parity_archive"]["evaluated_frames"], 0)
            self.assertEqual(len(sent), 1)

    def test_restore_exact_base_after_profiler_subclass_replacement(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        base = E.Run4EvaluatorV2
        previous = (base._evaluate_with, base.submit, base.close, base._observe_components)
        with tempfile.TemporaryDirectory() as root:
            try:
                with runtime.install_edge_evaluation(archive_directory=Path(root)/"p", max_frames=1):
                    class Profiled(base):
                        pass
                    E.Run4EvaluatorV2 = Profiled
                    self.assertIsNot(base.submit, previous[1])
                self.assertEqual((base._evaluate_with, base.submit, base.close, base._observe_components), previous)
            finally:
                E.Run4EvaluatorV2 = base


if __name__ == "__main__":
    unittest.main()

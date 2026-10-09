"""CPU injected replay, full-cohort failure accounting and retained joins."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.test_offline_quality_grid import reward_spec
from rl_agent.splitfusion_paired_quality_replay_v1 import test_capture as fixtures
from rl_agent.splitfusion_paired_quality_replay_v1 import scope
from . import analyze as A


def measurement(frame=105, *, undefined=False):
    value = {"frame_id": frame}
    for name in ("person", "vehicle"):
        value.update({f"loc_{name}_eligible_gt": 0 if undefined else 2,
            f"loc_{name}_tp": 0 if undefined else 1, f"loc_{name}_fp": 1,
            f"loc_{name}_fn": 0 if undefined else 1,
            f"loc_{name}_matched_xy_errors_m": [] if undefined else [.5],
            f"seg_{name}_gt_pixels": 4, f"seg_{name}_pred_pixels": 3,
            f"seg_{name}_intersection_pixels": 2, f"seg_{name}_union_pixels": 5})
    return value


class StatisticsTests(unittest.TestCase):
    def test_exact_quality_and_class_specific_statistics(self):
        spec, raw = reward_spec(), measurement()
        result = A.summarize_measurement(raw, spec)
        exact = A.Q.evaluate_exact_quality(spec, raw)
        self.assertEqual(result["quality"]["q_perc"], exact.q_perc)
        for name in ("vehicle", "person"):
            cls = result["classes"][name]
            self.assertEqual((cls["tp"], cls["fp"], cls["fn"], cls["recall"]), (1, 1, 1, .5))
            self.assertEqual(cls["matched_xy_error_median_m"], .5)
            self.assertEqual(cls["segmentation_surrogate"]["iou"], .4)

    def test_undefined_is_never_zero_quality(self):
        result = A.summarize_measurement(measurement(undefined=True), reward_spec())
        self.assertFalse(result["quality"]["defined"])
        self.assertIsNone(result["quality"]["q_perc"])
        self.assertIsNone(result["classes"]["vehicle"]["recall"])
        self.assertIsNone(result["classes"]["person"]["matched_xy_error_mean_m"])

    def test_missing_frame_never_shrinks_cohort_denominator(self):
        scored = A.summarize_measurement(measurement(), reward_spec())
        good = dict(status="PAIRED", selected=scored, dense=scored, delta_q_perc=0.)
        result = A.summarize_cohort([good, dict(status="JOIN_FAULT")])
        self.assertEqual(result["prepared_policy_inputs"], 2)
        self.assertEqual(result["paired_frames"], 1)
        self.assertFalse(result["full_cohort_complete"])
        self.assertIsNone(result["full_cohort_quality"])
        complete = A.summarize_cohort([good, good])
        self.assertEqual(complete["full_cohort_quality"]["selected_classes"]["vehicle"]["eligible_gt"], 4)
        self.assertEqual(complete["full_cohort_quality"]["selected_classes"]["vehicle"]["recall"], .5)
        self.assertIsNone(A.summarize_cohort([good], ["source drift"])["full_cohort_quality"])

    def test_defined_and_undefined_support_accounted_separately(self):
        yes = A.summarize_measurement(measurement(), reward_spec())
        no = A.summarize_measurement(measurement(undefined=True), reward_spec())
        rows = [dict(status="PAIRED", selected=yes, dense=yes, delta_q_perc=0.),
                dict(status="PAIRED", selected=no, dense=no, delta_q_perc=None)]
        result = A.summarize_cohort(rows)
        self.assertEqual(result["q_both_defined_frames"], 1)
        self.assertEqual(result["q_both_undefined_frames"], 1)
        self.assertEqual(result["full_cohort_quality"]["q_defined_denominator"], 1)
        self.assertEqual(result["full_cohort_quality"]["q_undefined_denominator"], 1)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case, planned, prepared = fixtures.fixture()
        archive = A.C.InputArchive(max_frames=1)
        record = archive.begin(planned)
        archive.observe_input(self.case.input, "cuda:0")
        archive.end(record, prepared)
        self.archive = self.root / "input"
        self.manifest = archive.flush(self.archive, collectors=fixtures.collectors(), source_binding=scope.PINS)
        self.spec = reward_spec()
        self.spec_path = self.root / "reward.json"
        with self.spec_path.open("xb") as stream:
            stream.write(self.spec.canonical_bytes())
        self.cell = self.root / "cell"
        self.cell.mkdir()
        self.pins = {}
        for name in A.REQUIRED_CELL_PINS:
            path = self.cell / name
            path.parent.mkdir(parents=True, exist_ok=True)
            A.C.write_json(path, {})
            self.pins[name] = A.C.sha(path)
        frame = self.manifest["frames"][0]
        from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry, camera_world_matrix
        context = self.case.context
        calibration = StaticCameraRegistry.audited().resolve(context.camera_model_sha256, context.camera_mount_sha256)
        self.loaded_cell = dict(sent={A._key(frame["identity"]): {"run4_identity": frame["identity"]}},
                                archived={A._key(frame["identity"]): {}}, ue={}, edge={})
        self.joined = dict(archived_selected=A.RP.Prediction(self.case.labels, [self.case.record]),
            prepared=object(), avo={}, original_evaluation=None, evidence={"original_evaluation_present": False},
            source_camera_matrix=camera_world_matrix(context, calibration).tolist())
        self.adapter = NS(replay_callbacks=lambda: dict(front=self.case.front, encode_selected=self.case.encode,
            decode_selected=self.case.decode, tail=self.case.tail))
        self.kwargs = dict(expected_input_manifest_sha256=A.C.sha(self.archive / "MANIFEST.json"),
            evidence_sha256=self.pins, reward_spec_path=self.spec_path,
            expected_reward_spec_sha256=self.spec.canonical_sha256(), model_factory=lambda: self.adapter)

    def evaluate(self, **kwargs):
        raw = measurement()
        return NS(measurement=raw, evidence={"matched_pairs": {"vehicle": [], "person": []},
                                            "scoring": {"segmentation": {"cpu_fixture": True}}})

    def test_injected_callbacks_use_real_sealed_parity_then_exact_quality(self):
        with patch.object(A, "load_cell", return_value=self.loaded_cell), \
                patch.object(A, "join_frame", return_value=self.joined), \
                patch.object(A.E, "evaluate_prepared", side_effect=self.evaluate):
            report = A.analyze(self.archive, self.cell, self.root / "out", **self.kwargs)
        self.assertTrue(report["summary"]["full_cohort_complete"])
        self.assertEqual(report["rows"][0]["status"], "PAIRED")
        self.assertEqual(self.case.calls, ["front", "encode", "decode", ("tail", 2.), ("tail", 3.)])
        self.assertEqual(report["rows"][0]["delta_q_perc"], 0.)
        manifest = A._json(self.root / "out/MANIFEST.json")
        self.assertEqual(set(manifest["sha256"]), {"ANALYSIS.json", "FRAMES.csv"})
        for name, digest in manifest["sha256"].items():
            self.assertEqual(A.C.sha(self.root / "out" / name), digest)
        with self.assertRaisesRegex(ValueError, "create-only"):
            A.analyze(self.archive, self.cell, self.root / "out", **self.kwargs)

    def test_join_failure_and_audit_do_not_construct_model(self):
        def forbidden():
            raise AssertionError("model loading forbidden")
        self.kwargs["model_factory"] = forbidden
        with patch.object(A, "load_cell", return_value=self.loaded_cell), \
                patch.object(A, "join_frame", side_effect=ValueError("MISSING_ARCHIVED_PREDICTION")):
            report = A.analyze(self.archive, self.cell, self.root / "missing", **self.kwargs)
        self.assertEqual(report["summary"]["prepared_policy_inputs"], 1)
        self.assertEqual(report["rows"][0]["status"], "JOIN_FAULT")
        with patch.object(A, "load_cell", return_value=self.loaded_cell), \
                patch.object(A, "join_frame", return_value=self.joined):
            audit = A.analyze(self.archive, self.cell, self.root / "audit", audit_only=True, **self.kwargs)
        self.assertEqual(audit["rows"][0]["status"], "JOINED_NOT_REPLAYED")
        self.assertIsNone(audit["summary"]["full_cohort_quality"])
        self.assertEqual(self.case.calls, [])

    def test_payload_parity_fault_refuses_dense_and_scoring(self):
        self.adapter.replay_callbacks = lambda: dict(front=self.case.front, encode_selected=lambda *_: b"wrong",
            decode_selected=self.case.decode, tail=self.case.tail)
        with patch.object(A, "load_cell", return_value=self.loaded_cell), \
                patch.object(A, "join_frame", return_value=self.joined), \
                patch.object(A.E, "evaluate_prepared", side_effect=AssertionError("scoring forbidden")):
            report = A.analyze(self.archive, self.cell, self.root / "parity", **self.kwargs)
        self.assertEqual(report["rows"][0]["status"], "PARITY_FAULT")
        self.assertEqual(self.case.calls, ["front"])
        self.assertIsNone(report["summary"]["full_cohort_quality"])

    def test_camera_pose_mismatch_refuses_model_loading(self):
        self.joined["source_camera_matrix"][0][3] += .01
        with patch.object(A, "load_cell", return_value=self.loaded_cell), \
                patch.object(A, "join_frame", return_value=self.joined):
            report = A.analyze(self.archive, self.cell, self.root / "pose", **self.kwargs)
        self.assertEqual(report["rows"][0]["status"], "JOIN_FAULT")
        self.assertIn("camera pose differs", report["rows"][0]["fault"])
        self.assertFalse(report["model"]["load_attempted"])
        self.assertEqual(self.case.calls, [])

    def test_outer_pin_fault_reports_every_prepared_input_without_model(self):
        with patch.object(A, "load_cell", side_effect=ValueError("evidence hash mismatch")):
            report = A.analyze(self.archive, self.cell, self.root / "pins", **self.kwargs)
        self.assertEqual(len(report["rows"]), self.manifest["policy_opportunities"])
        self.assertTrue(report["global_faults"])
        self.assertEqual(self.case.calls, [])

    def test_failed_model_factory_is_attempted_once_for_entire_cohort(self):
        manifest = deepcopy(self.manifest)
        manifest["frames"].append(dict(manifest["frames"][0], index=1))
        # Only the population seam is synthetic here; the test isolates failed
        # resource construction and does not attempt replay of duplicate rows.
        with patch.object(A.C, "verify_archive", return_value=manifest), \
                patch.object(A, "load_cell", return_value=self.loaded_cell), \
                patch.object(A, "join_frame", return_value=self.joined), \
                patch.dict(self.kwargs, model_factory=unittest.mock.Mock(side_effect=RuntimeError("load failed"))):
            report = A.analyze(self.archive, self.cell, self.root / "model_failure", **self.kwargs)
            self.kwargs["model_factory"].assert_called_once()
        self.assertEqual([r["status"] for r in report["rows"]], ["MODEL_UNAVAILABLE"] * 2)
        self.assertFalse(report["model"]["loaded"])
        self.assertEqual(len(report["global_faults"]), 1)


RETAINED_CELL = Path(os.environ.get("SPLITFUSION_RETAINED_TEST_CELL",
    "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/"
    "experiments/splitfusion_live_parity_runtime_v5/20261003_optimization_v5/qualification_100/capture/cells/a71__favorable_stable"))


class RetainedReadOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins = {name: A.C.sha(RETAINED_CELL / name) for name in A.REQUIRED_CELL_PINS}
        cls.cell = A.load_cell(RETAINED_CELL, cls.pins)
        saved = next(iter(cls.cell["archived"].values()))
        identity = saved["action_identity"]
        decision = cls.cell["decisions"][(identity["session_uuid"], identity["tensor_seq"], identity["frame_id"])]
        cls.frame = dict(identity=identity, gt_identity=saved["gt_identity"],
            carla_timestamp=decision["carla_timestamp"], radar_window_sha256=decision["radar_window_sha256"])

    def test_actual_selected_archive_complete_gt_avo_and_original_score_join(self):
        joined = A.join_frame(self.frame, self.cell)
        result = A.E.evaluate_prepared(identity=self.frame["identity"], prepared=joined["prepared"],
            predicted_labels=joined["archived_selected"].labels,
            predictions=joined["archived_selected"].records, avo=joined["avo"])
        self.assertEqual(result.measurement, joined["original_evaluation"]["measurement"])
        self.assertTrue(joined["evidence"]["original_evaluation_present"])
        self.assertGreater(joined["evidence"]["last_packet_send_raw_ns"], 0)

    def test_missing_transmission_prediction_or_target_is_explicit(self):
        for collection, reason in (("sent", "MISSING_TRANSMITTED_IDENTITY"),
                                   ("archived", "MISSING_ARCHIVED_PREDICTION"),
                                   ("targets", "MISSING_COMPLETE_TARGET")):
            with self.subTest(collection=collection):
                cell = dict(self.cell, **{collection: {}})
                with self.assertRaisesRegex(ValueError, reason):
                    A.join_frame(self.frame, cell)

    def test_radar_timestamp_and_lastsend_must_match(self):
        altered = dict(self.frame, radar_window_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "radar binding"):
            A.join_frame(altered, self.cell)
        altered = dict(self.frame, carla_timestamp=self.frame["carla_timestamp"] + .1)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            A.join_frame(altered, self.cell)
        ue = deepcopy(self.cell["ue"])
        ue["gt_last_datagram"]["marks"][str(self.frame["identity"]["frame_id"])] += 1
        with self.assertRaisesRegex(ValueError, "LASTSEND"):
            A.join_frame(self.frame, dict(self.cell, ue=ue))

    def test_foreign_outer_sha_refused(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            A.load_cell(RETAINED_CELL, dict(self.pins, **{A.UE: "0" * 64}))


if __name__ == "__main__":
    unittest.main()

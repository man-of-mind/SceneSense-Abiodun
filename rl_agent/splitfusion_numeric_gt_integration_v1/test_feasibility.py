"""Correctness-only tests; elapsed test time is not a performance measurement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import feasibility as f


class ArithmeticTests(unittest.TestCase):
    def test_gate_and_fixed_spans_use_integer_nanoseconds(self):
        row = f.accounting(action=0, compute_start=20, compute_end=50, submit=60,
                           admitted=80, evaluate_start=90, emit_start=110, emit_end=115,
                           receipt=120)
        self.assertEqual(row["gt_critical_wait_ns"], 20)
        self.assertEqual(row["reconstructed_feedback_ns"], 120)
        self.assertEqual(row["reconstruction_error_ns"], 0)
        self.assertEqual(row["gt_ready_by_submit_fixed_residual_ns"], 100)
        self.assertEqual(row["ideal_gt_zero_bridge_floor_ns"], 55)
        self.assertEqual(row["ideal_gt_bridge_5ms_ns"], 5_000_055)

    def test_prediction_can_be_the_later_gate(self):
        row = f.accounting(action=0, compute_start=20, compute_end=50, submit=60,
                           admitted=30, evaluate_start=90, emit_start=110, emit_end=115,
                           receipt=120)
        self.assertFalse(row["gt_later_than_evaluator_submit"])
        self.assertEqual(row["gt_critical_wait_ns"], 0)
        self.assertEqual(row["gt_ready_by_submit_fixed_residual_ns"], 120)

    def test_inclusive_deadline_and_linear_quantiles(self):
        stats = f.stats_ns([169_999_999, 170_000_000, 170_000_001])
        self.assertEqual(stats["inclusive_170ms_count"], 2)
        self.assertEqual(stats["p50"], 170)
        self.assertEqual(f.stats_ns([10_000_000, 20_000_000])["p95"], 19.5)

    def test_clock_and_noncausal_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "CLOCK_MONOTONIC_RAW"):
            f.raw_stamp({"domain": "CLOCK_REALTIME", "ns": 100})
        with self.assertRaisesRegex(ValueError, "noncausal"):
            f.accounting(action=0, compute_start=20, compute_end=70, submit=60,
                         admitted=80, evaluate_start=90, emit_start=110, emit_end=115,
                         receipt=120)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            f.index([{"frame_id": 1}, {"frame_id": 1}], "test")

    def test_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            f.write_results(output, [{"frame_id": 1}], {"status": f.LABEL})
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                f.write_results(output, [{"frame_id": 2}], {"status": "bad"})
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


class RetainedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        comparison = json.loads((f.DEFAULT_TRIAL / "comparison/RESULT.json").read_text())
        cell = f.DEFAULT_TRIAL / "treatment/capture/cells/a71__favorable_stable"
        cls.inputs = ["treatment", comparison["arms"]["treatment"]]
        for name in ("run4_phase6/PHASE6_UE_EVIDENCE.json", "run4_phase6/edge_report.json",
                     "CPU_GT_PROCESS_TIMELINE.json", "profiler/edge/EDGE_HANDOFF_SCOPE.json"):
            cls.inputs.append(json.loads((cell / name).read_text()))
        for name in ("profiler/ue.profile.jsonl", "profiler/edge/edge.profile.jsonl"):
            cls.inputs.append([json.loads(line) for line in (cell / name).read_text().splitlines()])

    def test_all_97_frames_and_undefined_replies_remain(self):
        rows, summary = f.analyze_trial()
        self.assertEqual(len(rows), 97)
        treatment, control = summary["arms"]["treatment"], summary["arms"]["control"]
        self.assertEqual((treatment["policy_frames"], control["policy_frames"]), (49, 48))
        self.assertEqual((treatment["undefined_quality_frames"], control["undefined_quality_frames"]), (8, 1))
        self.assertEqual(treatment["observed_feedback_ms"]["p50"], 257.441293)
        self.assertEqual(treatment["gt_ready_by_submit_fixed_residual_ms"]["p50"], 197.221716)
        self.assertEqual(treatment["gt_ready_by_submit_fixed_residual_ms"]["inclusive_170ms_count"], 2)
        self.assertEqual(treatment["gt_later_than_submit_count"], 49)
        self.assertEqual([treatment["ideal_gt_bridge_scenarios_ms"][str(b)]["inclusive_170ms_count"]
                          for b in f.BUDGETS_MS], [31, 27, 24, 19, 15])
        self.assertEqual([control["ideal_gt_bridge_scenarios_ms"][str(b)]["inclusive_170ms_count"]
                          for b in f.BUDGETS_MS], [33, 28, 24, 21, 14])
        self.assertTrue(all(row["reconstruction_error_ns"] == 0 for row in rows))
        self.assertEqual(sum(row["quality_undefined"] for row in rows), 9)
        self.assertFalse(summary["measured_intervention"])
        self.assertFalse(summary["deadline_qualification_claim"])

    def test_missing_feedback_is_rejected_instead_of_dropped(self):
        args = list(self.inputs)
        args[2] = dict(args[2], feedback_rows=args[2]["feedback_rows"][:-1])
        with self.assertRaisesRegex(ValueError, "missing/foreign policy frames: feedback"):
            f.analyze_data(*args)

    def test_foreign_session_in_profile_is_rejected(self):
        args = list(self.inputs)
        args[7] = copy.deepcopy(args[7])
        row = next(row for row in args[7] if row.get("stage") == "edge.compute"
                   and row.get("ph") == "live" and row.get("reward_requested") is True
                   and row["r"] == "span")
        row["session_uuid"] = "foreign-session"
        with self.assertRaisesRegex(ValueError, "profiler identity differs"):
            f.analyze_data(*args)

    def test_different_boot_is_rejected(self):
        args = list(self.inputs)
        args[6] = copy.deepcopy(args[6])
        next(row for row in args[6] if row["r"] == "header")["boot_id"] = "other-boot"
        with self.assertRaisesRegex(ValueError, "shared CLOCK_MONOTONIC_RAW boot"):
            f.analyze_data(*args)

    def test_duplicate_policy_frame_is_rejected(self):
        args = list(self.inputs)
        duplicate = next(row for row in args[2]["decisions"] if row["kind"] == "POLICY_DECISION")
        args[2] = dict(args[2], decisions=args[2]["decisions"] + [duplicate])
        with self.assertRaisesRegex(ValueError, "duplicate/invalid frame: policy"):
            f.analyze_data(*args)


if __name__ == "__main__":
    unittest.main()

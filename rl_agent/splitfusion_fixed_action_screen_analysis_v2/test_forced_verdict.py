"""Offline tests of the amendment-01 forced-screen verdict on real and deliberately failing evidence."""
import copy
import json
from pathlib import Path
import unittest

from .analyze import Arm, MAIN, WT, integration
from .forced_verdict import collect, evaluate

SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
ARM0 = SCREEN / "20261005_fixed_action_screen_v2/arm0_a33"
V1_ARM0 = SCREEN / "20261005_fixed_action_screen_v1/arm0_a33"
E = MAIN / "experiments"


class ForcedVerdictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integ = integration(Arm(ARM0), None, do_rescore=True)
        cls.base = collect(ARM0, launch_returncode=1,
                           launch_log=E / "splitfusion_fixed_action_screen_v2/campaign_02/launch_arm0_a33.log",
                           integration_result=integ)

    def mutated(self, fn):
        e = copy.deepcopy(self.base)
        fn(e)
        return evaluate(e)

    def stops(self, fn, condition):
        result = self.mutated(fn)
        self.assertEqual(result["verdict"], "STOP")
        self.assertTrue(any(r.startswith(condition) for r in result["reasons"]), result["reasons"])

    def test_real_arm0_passes_amended_rule(self):
        result = evaluate(self.base)
        self.assertEqual(result["verdict"], "PASS_FORCED_SCREEN", result)
        self.assertEqual((result["retained_verdict"], result["retained_g8"]), ("CAPTURE_INVALID", False))

    def test_real_crashed_v1_campaign_stops(self):
        integ = json.loads((E / "splitfusion_fixed_action_screen_v1/campaign_01/check_arm0_a33.json").read_text())
        e = collect(V1_ARM0, launch_returncode=1, launch_log=E / "splitfusion_fixed_action_screen_v1/campaign_01/launch_arm0_a33.log",
                    integration_result=integ)
        result = evaluate(e)
        self.assertEqual(result["verdict"], "STOP")
        for c in ("R", "G", "I"):
            self.assertFalse(result["conditions"][c])

    def test_missing_results_stop(self):
        self.assertEqual(self.mutated(lambda e: e.update(qualification=None))["reasons"], ["MISSING_RESULTS"])
        self.assertEqual(self.mutated(lambda e: e.update(cell_result=None))["reasons"], ["MISSING_RESULTS"])
        self.assertEqual(self.mutated(lambda e: e.update(semantics=None))["reasons"], ["MISSING_RESULTS"])

    def test_route_failures_stop(self):
        self.stops(lambda e: e["cell_result"]["child"].update(return_code=2), "R:child_return_code")
        self.stops(lambda e: e["cell_result"].update(status="FAILED"), "R:cell_passed")
        self.stops(lambda e: e["cell_result"]["child"]["collector"].update(collector_failures=["frame 1: crash"]),
                   "R:no_collector_failures")
        self.stops(lambda e: e["cell_result"]["child"]["collector"].update(stop_reason=""), "R:collector_stop_matches_g1")
        self.stops(lambda e: e["cell_result"]["child"]["route_detail"].update(error="BrokenPipeError: [Errno 32] Broken pipe"),
                   "R:route_error_is_retained_budget_stop")
        self.stops(lambda e: e["cell_result"]["child"]["cleanup"].update(map_stopped=False), "R:child_cleanup")
        self.stops(lambda e: e["cell_result"]["cleanup"].update(all_gates_passed=False), "R:cell_cleanup")

    def test_additional_gate_failures_stop(self):
        for gate in ("G1_FRAME_ACCOUNTING", "G3_ZERO_IDENTITY_CONFLICTS", "G6_NO_PROFILER_INDUCED_FAULT", "G9_HOST_RESTORED_CLEAN"):
            self.stops(lambda e, g=gate: e["qualification"]["gates"].update({g: False}), "G:other_gates_true")
        self.stops(lambda e: e["qualification"]["gates"].update(G10_NEW=True), "G:no_unknown_gates")

    def test_other_g8_components_must_pass(self):
        self.stops(lambda e: e["semantics"]["reward"].update(passed=False), "C:reward_semantics")
        self.stops(lambda e: e["semantics"]["receipt_marks"].update(passed=False), "C:receipt_marks")
        self.stops(lambda e: e["qualification"]["phase6_report_only"]["gates"].update(P4_ACCOUNTING=False), "C:p3_p4_p5")

    def test_actor_replay_failure_must_be_only_forced_comparison(self):
        def bad_replay(e):
            m = e["semantics"]["actor_replay"]["mismatch_examples"][0]
            m["replayed"] = [m["replayed"][0], m["replayed"][1] + 1] + m["replayed"][2:]
        self.stops(bad_replay, "A:only_forced_comparison")
        self.stops(lambda e: e["semantics"]["actor_replay"]["mismatch_examples"][0].update(planned=[0, 0]),
                   "A:only_forced_comparison")
        self.stops(lambda e: e["semantics"]["actor_replay"].update(replayed=1), "A:actor_replay_reported")

    def test_forced_execution_and_independent_replay(self):
        self.stops(lambda e: e["forced_replay"].update(mismatches=1), "F:independent_replay_zero_mismatch")
        self.stops(lambda e: e["integration"]["checks"].update(I3_forced_action=False), "F:I3_forced_action")

    def test_remaining_integration_criteria(self):
        for name in ("I4_terminal_accounting", "I5_exact_quality_offline", "I8_bounded", "I9_cleanup", "I11_anchor_provenance"):
            self.stops(lambda e, n=name: e["integration"]["checks"].update({n: False}), "I:" + name)
        self.stops(lambda e: e["integration"].update(launch_returncode=1), "I:integration_evaluated_without_launch_clause")

    def test_exit_codes(self):
        self.stops(lambda e: e.update(launch_returncode=2), "X:unknown_exit_code_2")
        self.stops(lambda e: e.update(launch_returncode=-9), "X:unknown_exit_code_-9")
        self.stops(lambda e: e.update(launch_returncode=None), "X:unknown_exit_code_None")
        self.stops(lambda e: e.update(launch_log_tail=e["launch_log_tail"] + "\nValueError: something else\n"),
                   "X:expected_exception_only")
        self.stops(lambda e: e.update(launch_log_tail=""), "X:expected_exception_only")
        self.stops(lambda e: e["qualification"].update(verdict="CAPTURE_VALID"), "X:retained_verdict_invalid")


if __name__ == "__main__":
    unittest.main()

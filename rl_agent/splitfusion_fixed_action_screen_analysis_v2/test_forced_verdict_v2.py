"""Offline tests of the amendment-02 verdict: both permitted exceptions accepted, unrelated failures stop."""
import copy
import json
import unittest

from .analyze import Arm, MAIN, WT, integration
from .forced_verdict_v2 import collect, evaluate

SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
E = MAIN / "experiments"


def load(arm_name, campaign, launch_log):
    arm = SCREEN / "20261005_fixed_action_screen_v2" / arm_name
    return collect(arm, launch_returncode=1, launch_log=E / "splitfusion_fixed_action_screen_v2" / campaign / launch_log,
                   integration_result=integration(Arm(arm), None, do_rescore=True))


class AmendmentTwoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm0 = load("arm0_a33", "campaign_02", "launch_arm0_a33.log")
        cls.arm1 = load("arm1_a65", "campaign_03", "launch_arm1_a65.log")

    def check(self, base, fn, condition):
        e = copy.deepcopy(base)
        fn(e)
        result = evaluate(e)
        self.assertEqual(result["verdict"], "STOP", result)
        self.assertTrue(any(r.startswith(condition) for r in result["reasons"]), result["reasons"])

    def test_both_permitted_exceptions_accepted(self):
        r1 = evaluate(self.arm1)                         # G8 actor-replay mismatch AND P0 coverage false
        self.assertEqual(r1["verdict"], "PASS_FORCED_SCREEN", r1)
        self.assertEqual((r1["p0_policy_coverage"], r1["cell_status"], r1["retained_g8"]), (False, "INCONCLUSIVE_OR_FAILED", False))
        r0 = evaluate(self.arm0)                         # only the G8 exception
        self.assertEqual(r0["verdict"], "PASS_FORCED_SCREEN", r0)
        self.assertEqual(r0["cell_status"], "PASSED")

    def test_real_crashed_v1_campaign_still_stops(self):
        integ = json.loads((E / "splitfusion_fixed_action_screen_v1/campaign_01/check_arm0_a33.json").read_text())
        e = collect(SCREEN / "20261005_fixed_action_screen_v1/arm0_a33", launch_returncode=1,
                    launch_log=E / "splitfusion_fixed_action_screen_v1/campaign_01/launch_arm0_a33.log", integration_result=integ)
        self.assertEqual(evaluate(e)["verdict"], "STOP")

    def test_other_phase6_gates_still_bind(self):
        for gate in ("P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING", "P5_NO_ACTOR_AFTER_REFUSAL", "P6_RESTORE_COLD",
                     "P7_FEEDBACK_OVER_DOWNLINK", "P8_NO_INFRASTRUCTURE_FAULT"):
            self.check(self.arm1, lambda e, g=gate: e["qualification"]["phase6_report_only"]["gates"].update({g: False}),
                       "R:phase6_binding_gates_true")
        self.check(self.arm1, lambda e: e["qualification"]["phase6_report_only"]["gates"].update(P9_NEW=True),
                   "R:phase6_gate_set_known")

    def test_status_must_match_gates(self):
        self.check(self.arm1, lambda e: e["cell_result"].update(status="FAILED"), "R:cell_status_consistent_with_gates")
        self.check(self.arm0, lambda e: e["cell_result"].update(status="INCONCLUSIVE_OR_FAILED"),
                   "R:cell_status_consistent_with_gates")
        self.check(self.arm1, lambda e: e["qualification"]["phase6_report_only"].update(status="PASSED"),
                   "R:cell_status_consistent_with_gates")

    def test_unrelated_failures_still_stop(self):
        b = self.arm1
        self.check(b, lambda e: e["cell_result"].update(error="ChildError: crash"), "R:no_cell_error")
        self.check(b, lambda e: e["cell_result"]["child"].update(error="boom"), "R:no_child_error")
        self.check(b, lambda e: e["cell_result"]["child"].update(return_code=2), "R:child_return_code")
        self.check(b, lambda e: e["cell_result"]["child"]["collector"].update(collector_failures=["x"]), "R:no_collector_failures")
        self.check(b, lambda e: e["cell_result"]["child"]["route_detail"].update(error="BrokenPipeError: x"),
                   "R:route_error_is_retained_budget_stop")
        self.check(b, lambda e: e["cell_result"]["cleanup"].update(all_gates_passed=False), "R:cell_cleanup")
        self.check(b, lambda e: e["qualification"]["gates"].update(G3_ZERO_IDENTITY_CONFLICTS=False), "G:other_gates_true")
        self.check(b, lambda e: e["qualification"]["gates"].update(G9_HOST_RESTORED_CLEAN=False), "G:other_gates_true")
        self.check(b, lambda e: e["semantics"]["reward"].update(passed=False), "C:reward_semantics")
        self.check(b, lambda e: e["semantics"]["receipt_marks"].update(passed=False), "C:receipt_marks")
        self.check(b, lambda e: e["semantics"]["actor_replay"]["mismatch_examples"][0].update(planned=[0, 0]),
                   "A:only_forced_comparison")
        self.check(b, lambda e: e["forced_replay"].update(mismatches=1), "F:independent_replay_zero_mismatch")
        for name in ("I3_forced_action", "I4_terminal_accounting", "I5_exact_quality_offline", "I6_history", "I8_bounded",
                     "I9_cleanup", "I11_anchor_provenance"):
            self.check(b, lambda e, n=name: e["integration"]["checks"].update({n: False}), "")
        self.check(b, lambda e: e.update(launch_returncode=2), "X:unknown_exit_code_2")
        self.check(b, lambda e: e.update(launch_log_tail="ValueError: other\n"), "X:expected_exception_only")
        e = copy.deepcopy(b); e["qualification"] = None
        self.assertEqual(evaluate(e)["verdict"], "STOP")


if __name__ == "__main__":
    unittest.main()

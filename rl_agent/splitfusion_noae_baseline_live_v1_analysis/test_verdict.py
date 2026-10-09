"""Offline tests of the registered baseline verdict on retained real forced evidence (v2 arm 4, action 67; read-only).

The retained arm ran the ORIGINAL Run-5 actor in shadow, so its integration is computed here with the retained v3
replay; the full-actor substitution is tested separately (it must reproduce none of those proposals and restore).
Run from the main checkout with CUDA_VISIBLE_DEVICES=''.
"""
import copy
import unittest

from rl_agent.splitfusion_fixed_action_screen_analysis_v2 import analyze as A2
from rl_agent.splitfusion_fixed_action_screen_analysis_v3 import integration as I3, verdict as V3
from rl_agent.splitfusion_fixed_action_screen_analysis_v3.test_verdict import ARM, LOG
from rl_agent.splitfusion_fixed_action_screen_v3 import contract as C3

from . import verdict as V

GC_PASS = dict(passed=True)


class BaselineVerdictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resolved = C3.resolve(11, 3000)
        arm = A2.Arm(ARM)
        arm.forced = dict(resolved)
        base = I3.integration(arm, None, do_rescore=True)
        i8a, i8b = V.split_i8(arm)
        checks = dict(base["checks"])
        checks.pop("I8_bounded")
        checks["I8a_gt_service_within_limits"] = i8a
        cls.integ = dict(base, checks=checks, i8b_reported_outcomes=i8b)
        cls.evidence = V3.collect(ARM, launch_returncode=1, launch_log=LOG, integration_result=cls.integ)
        cls.evidence["forced_action"] = dict(resolved)

    def test_retained_valid_forced_arm_is_valid(self):
        out = V.evaluate(self.evidence, self.integ, GC_PASS)
        self.assertEqual(out["verdict"], "VALID_BASELINE_MEASUREMENT", out["reasons"])

    def test_gc_or_producer_failure_stops(self):
        self.assertEqual(V.evaluate(self.evidence, self.integ, dict(passed=False))["verdict"], "STOP_INVALID")

    def test_refusals_and_queue_overflow_are_reported_not_invalidating(self):
        e = copy.deepcopy(self.evidence)
        e["integration"]["checks"].pop("I8_bounded", None)
        out = V.evaluate(e, self.integ, GC_PASS)
        self.assertEqual(out["verdict"], "VALID_BASELINE_MEASUREMENT")
        self.assertIn("gt_service_refusals", out["reported_outcomes"]["i8b"])

    def test_gt_service_occupancy_breach_stops(self):
        integ = copy.deepcopy(self.integ)
        integ["checks"]["I8a_gt_service_within_limits"] = False
        e = dict(self.evidence, integration=integ)
        self.assertEqual(V.evaluate(e, integ, GC_PASS)["verdict"], "STOP_INVALID")

    def test_infrastructure_fault_and_identity_failures_stop(self):
        for gate in ("P8_NO_INFRASTRUCTURE_FAULT", "P1_IDENTITY", "P7_FEEDBACK_OVER_DOWNLINK"):
            e = copy.deepcopy(self.evidence)
            e["qualification"]["phase6_report_only"]["gates"][gate] = False
            self.assertEqual(V.evaluate(e, self.integ, GC_PASS)["verdict"], "STOP_INVALID", gate)
        e = copy.deepcopy(self.evidence)
        e["qualification"]["gates"]["G3_ZERO_IDENTITY_CONFLICTS"] = False
        self.assertEqual(V.evaluate(e, self.integ, GC_PASS)["verdict"], "STOP_INVALID")

    def test_p0_report_only(self):
        e = copy.deepcopy(self.evidence)
        self.assertTrue(V.evaluate(e, self.integ, GC_PASS)["verdict"] == "VALID_BASELINE_MEASUREMENT")
        self.assertIn("p0_policy_coverage", V.evaluate(e, self.integ, GC_PASS)["reported_outcomes"])

    def test_exit_code_alone_never_decides(self):
        self.assertEqual(V.evaluate(dict(self.evidence, launch_returncode=2), self.integ, GC_PASS)["verdict"], "STOP_INVALID")
        self.assertEqual(V.evaluate(dict(self.evidence, qualification=None), self.integ, GC_PASS)["verdict"], "STOP_INVALID")

    def test_full_actor_replay_substitution_is_scoped_and_uses_the_full_actor(self):
        original = A2.forced_replay
        with V.full_actor_replay():
            self.assertIs(A2.forced_replay, V.forced_replay_full)
        self.assertIs(A2.forced_replay, original)
        arm = A2.Arm(ARM)
        decisions = {r["frame_id"]: r["executed"] for r in A2.load_jsonl(arm.cell / "forced_action/FORCED_ACTIONS.jsonl")
                     if r["kind"] == "POLICY_DECISION"}
        replay = V.forced_replay_full(arm, decisions)
        self.assertEqual(replay["actor"], "FULL_RUN5_SEED43_U10000")
        self.assertGreater(replay["replayed"], 0)
        self.assertEqual(replay["mismatches"], replay["replayed"])   # that arm's shadow actor was the original Run-5 actor


if __name__ == "__main__":
    unittest.main()

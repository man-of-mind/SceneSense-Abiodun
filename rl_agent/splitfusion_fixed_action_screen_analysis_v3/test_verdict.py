"""Offline tests of the v3 integration/verdict on retained real evidence (v2 arm 4, action 67; read-only).

Run from the main checkout with CUDA_VISIBLE_DEVICES='' (exact re-scoring of every scored decision).
"""
import copy
import unittest

from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import WT, MAIN
from rl_agent.splitfusion_fixed_action_screen_v3 import contract as C

from .integration import Arm, integration
from . import verdict as V

ARM = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_screen_v2/arm4_a67"
LOG = MAIN / "experiments/splitfusion_fixed_action_screen_v2/campaign_04/launch_arm4_a67.log"


class RealArmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolved = C.resolve(11, 3000)
        arm = Arm(ARM)
        arm.forced = dict(cls.resolved)
        cls.integ = integration(arm, None, do_rescore=True)
        cls.evidence = V.collect(ARM, launch_returncode=1, launch_log=LOG, integration_result=cls.integ)
        cls.evidence["forced_action"] = dict(cls.resolved)

    def test_real_anchor_arm_passes_strict_checks(self):
        self.assertTrue(self.integ["checks"]["I11_anchor_provenance"])
        self.assertTrue(self.integ["checks"]["I12_executed_identity"])
        self.assertTrue(self.integ["passed"], self.integ["checks"])
        self.assertEqual(V.evaluate(self.evidence)["verdict"], "PASS_FORCED_SCREEN")

    def test_offanchor_expectation_on_anchor_evidence_stops(self):
        arm = Arm(ARM)
        arm.forced = dict(self.resolved, action_id=None, profile_id=None, measurement_status="UNMEASURED_OFF_ANCHOR")
        integ = integration(arm, None, do_rescore=False)
        self.assertFalse(integ["checks"]["I11_anchor_provenance"])
        self.assertFalse(integ["checks"]["I12_executed_identity"])
        e = dict(self.evidence, integration=dict(self.integ, checks=integ["checks"]))
        self.assertEqual(V.evaluate(e)["verdict"], "STOP")

    def test_wrong_bundle_stops(self):
        arm = Arm(ARM)
        arm.forced = dict(self.resolved, execution_bundle_sha256="0" * 64)
        integ = integration(arm, None, do_rescore=False)
        self.assertFalse(integ["checks"]["I12_executed_identity"])
        e = dict(self.evidence, integration=dict(self.integ, checks=integ["checks"]))
        self.assertIn("I:I12_executed_identity", V.evaluate(e)["reasons"])

    def test_unknown_exit_code_and_missing_results_stop(self):
        self.assertEqual(V.evaluate(dict(self.evidence, launch_returncode=2))["verdict"], "STOP")
        self.assertEqual(V.evaluate(dict(self.evidence, qualification=None))["verdict"], "STOP")

    def test_extra_exception_in_launch_log_stops(self):
        e = dict(self.evidence, launch_log_tail=self.evidence["launch_log_tail"] + "\nValueError: something else\n")
        self.assertEqual(V.evaluate(e)["verdict"], "STOP")

    def test_other_gate_failure_stops(self):
        e = copy.deepcopy(self.evidence)
        e["qualification"]["gates"]["G3_ZERO_IDENTITY_CONFLICTS"] = False
        self.assertEqual(V.evaluate(e)["verdict"], "STOP")

    def test_p0_report_only_and_other_phase6_gate_stops(self):
        e = copy.deepcopy(self.evidence)
        e["qualification"]["phase6_report_only"]["gates"]["P7_FEEDBACK_OVER_DOWNLINK"] = False
        self.assertEqual(V.evaluate(e)["verdict"], "STOP")


if __name__ == "__main__":
    unittest.main()

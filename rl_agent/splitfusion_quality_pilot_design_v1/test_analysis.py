"""CPU-only tests for the quality/pilot design analysis. Run from abiodun/ with env -u PYTHONPATH."""
import json
import math
import unittest

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract
from . import analysis as A


def screen_row(*, receipt, q=0.5, v_el=0, v_err=(), p_el=0, p_err=()):
    return {"ue_receipt": receipt, "q_perc": q, "vehicle_eligible_gt": v_el, "vehicle_matched_xy_errors_m": list(v_err),
            "person_eligible_gt": p_el, "person_matched_xy_errors_m": list(p_err)}


class Definitions(unittest.TestCase):
    def test_tolerances_and_weights_equal_the_approved_spec(self):
        spec = json.loads((A.ROOT / "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
                           "20260918_exact_continuous_q_grid_a1b_full/reward_spec.json").read_text())
        self.assertEqual(A.TOL, {"vehicle": spec["tau_vehicle_m"], "person": spec["tau_person_m"]})
        self.assertEqual(A.WEIGHT, {"vehicle": spec["w_loc_vehicle"], "person": spec["w_loc_person"]})

    def test_current_reward_constants_are_the_run4_contract(self):
        self.assertEqual(A.DEADLINE_MS, run4_contract.REWARD_DEADLINE_MS)
        self.assertEqual(run4_contract.REWARD_LATENCY_WEIGHT, 0.25)
        self.assertEqual(A.current_reward(screen_row(receipt=170.0, q=0.4)), 0.4 - 0.25)
        self.assertEqual(A.current_reward(screen_row(receipt=170.001)), run4_contract.REGISTERED_FAILURE_REWARD)
        self.assertIsNone(A.current_reward(screen_row(receipt=100.0, q=None)))

    def test_within_tolerance_recall_counts_only_matches_inside_tolerance(self):
        # two of four vehicles matched, one beyond 1.0 m: R_tol = 1/4 although the median error is 0.75 m
        row = screen_row(receipt=150, v_el=4, v_err=(0.5, 1.0001))
        self.assertAlmostEqual(A.screen_y(row), 0.25)

    def test_arithmetic_combiner_does_not_zero_vehicle_success_when_persons_are_missed(self):
        row = screen_row(receipt=150, v_el=1, v_err=(0.2,), p_el=2, p_err=())
        self.assertAlmostEqual(A.screen_y(row), 0.4 * 1.0 / (0.4 + 0.6))

    def test_undefined_without_eligible_objects_and_late_is_zero(self):
        self.assertIsNone(A.otr(screen_row(receipt=100)))
        self.assertEqual(A.otr(screen_row(receipt=170.5, v_el=1, v_err=(0.1,))), 0.0)
        self.assertEqual(A.otr(screen_row(receipt=170.0, v_el=1, v_err=(0.1,))), 1.0)
        self.assertEqual(A.otr(screen_row(receipt=165.0, v_el=1, v_err=(0.1,)), shift_ms=10), 0.0)

    def test_wilson_matches_the_addendum_values(self):
        lo, hi = A.wilson(27, 48)
        self.assertEqual((round(lo, 2), round(hi, 2)), (0.42, 0.69))
        lo, hi = A.wilson(27, 46)
        self.assertEqual((round(lo, 2), round(hi, 2)), (0.44, 0.72))


class RecordedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.screen = A.load_screen()
        cls.warm = [x for x in cls.screen if not x["cold"]]

    def test_every_recorded_q_perc_reproduces_exactly_with_the_pinned_formula(self):
        r = A.recompute_q_perc(self.screen)
        self.assertEqual(r["definedness_mismatches"], 0)
        self.assertEqual(r["compared"], 312)
        self.assertEqual(r["max_abs_difference"], 0.0)

    def test_warm_counts_match_the_sealed_assessment(self):
        self.assertEqual(len(self.screen), 334)
        self.assertEqual(len(self.warm), 320)

    def test_scene_coverage_reproduces_253_of_512(self):
        support = {a: [x for x in self.warm if x["executed_action"]["action_id"] == a] for a in A.ACTIONS}
        scenes, p40 = A.load_grid()
        cov = A.coverage(scenes, p40, support)
        self.assertEqual(sum(1 for k in cov if k[0] == "fit"), 512)
        self.assertEqual(sum(1 for k, v in cov.items() if v and k[0] == "fit"), 253)
        self.assertEqual(sum(1 for k, v in cov.items() if v and k[0] == "held_scene"), 103)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the read-only held-scene diagnosis (synthetic inputs; CPU only)."""

from __future__ import annotations

import math
import unittest

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_diagnosis as D


def success(q, latency):
    return {"terminal": "SUCCESS", "delivered_q_perc": q, "latency_ms": latency,
            "reward": q - 0.25 * (latency / 170.0)}


TIMEOUT = {"terminal": "TIMEOUT", "delivered_q_perc": None, "latency_ms": None, "reward": -1.0}


class DiagnosisTest(unittest.TestCase):
    def test_reward_decomposition_is_exact_and_has_no_switch_term(self) -> None:
        rows = [success(0.6, 85.0), success(0.4, 150.0), TIMEOUT, success(0.55, 10.0)]
        out = D.decompose(rows)
        self.assertEqual(out["rows_reproduced_exactly"], 4)
        self.assertEqual(out["action_switch_penalty"], 0.0)
        self.assertLess(out["abs_residual"], 1e-15)
        self.assertAlmostEqual(out["timeout_or_failure"], -0.25)
        self.assertAlmostEqual(out["delivered_quality"], (0.6 + 0.4 + 0.55) / 4)

    def test_mode_q_regret_is_additive(self) -> None:
        def anchor(e):
            return {"in_support": True, "expected": e}
        scored = [{
            "row": {"mode_id": 1, "q_e4": 4200},
            "own": {"expected": 0.10},
            "fixed": anchor(0.12),
            "anchors": {(1, 3000): anchor(0.15), (1, 5000): anchor(0.11),
                        (2, 5000): anchor(0.30), (2, 3000): anchor(0.05),
                        (0, 0): {"in_support": False}},
        }]
        out = D.mode_q_regret(scored)
        self.assertAlmostEqual(out["within_mode_q_opportunity_mean"], 0.05)
        self.assertAlmostEqual(out["discrete_mode_opportunity_mean"], 0.15)
        self.assertAlmostEqual(out["total_restricted_regret_mean"], 0.20)
        self.assertEqual(out["additivity_residual"], 0.0)
        self.assertEqual(out["signed_q_minus_best_within_mode_anchor_mean"], 1200)
        self.assertEqual(out["best_within_mode_q_is_lower_fraction"], 1.0)
        # nearest registered q to 4200 is 5000: best mode there is 2 (0.30) vs own mode 1 (0.11)
        self.assertAlmostEqual(out["best_mode_at_nearest_registered_q_opportunity_mean"], 0.19)
        self.assertIn("ONE_STEP_RESTRICTED_ANCHOR", out["label"])

    def test_spearman(self) -> None:
        self.assertAlmostEqual(D.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(D.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isnan(D.spearman([1, 1], [2, 2])))      # all ties: undefined
        self.assertAlmostEqual(D.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)

    def test_outputs_are_verified_against_the_sealed_manifest(self) -> None:
        manifest = D.verify_outputs()
        self.assertEqual(len(manifest["files"]), 14)


if __name__ == "__main__":
    unittest.main()

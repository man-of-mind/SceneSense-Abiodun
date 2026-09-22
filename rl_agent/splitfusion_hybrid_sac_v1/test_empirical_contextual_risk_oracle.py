"""Focused tests for the smooth latency-percentile oracle hypotheses."""

from __future__ import annotations

import math
import unittest

import numpy as np

from .empirical_contextual_contract import PILOT_UTILITY_SPEC
from .empirical_contextual_risk_oracle import (
    BUDGET_MS,
    RISK_VARIANTS,
    SMOOTH_RISK_SPEC_SHA256,
    SPECS,
    _best_index,
    _require_registered_suite,
    _reward_vector,
)
class SmoothRiskSpecTests(unittest.TestCase):
    def test_registered_suite_hash_is_pinned(self) -> None:
        _require_registered_suite()
        self.assertEqual(len(SMOOTH_RISK_SPEC_SHA256), 64)

    def test_variants_are_exactly_control_p95_p99(self) -> None:
        self.assertEqual(RISK_VARIANTS, ("p50", "p95", "p99"))
        self.assertEqual(tuple(spec.percentile for spec in SPECS), RISK_VARIANTS)

    def test_p50_control_is_exactly_d1_for_scalar_inputs(self) -> None:
        spec = SPECS[0]
        observed = spec.expected_utility(
            p_edge_admission_given_sent=0.73,
            q_perc=0.61,
            latency_ms=187.25,
        )
        expected = PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=0.73,
            q_perc=0.61,
            latency_proxy_ms=187.25,
        )
        self.assertEqual(observed, expected)

    def test_vector_matches_scalar_for_every_variant(self) -> None:
        p = np.asarray([0.0, 0.25, 0.8, 1.0], dtype=np.float64)
        q = np.asarray([0.2, 0.4, 0.6, 0.9], dtype=np.float64)
        latency = np.asarray([150.0, 199.9, 220.0, 301.0], dtype=np.float64)
        for spec in SPECS:
            vector = _reward_vector(spec, p, q, latency)
            scalar = np.asarray(
                [
                    spec.expected_utility(
                        p_edge_admission_given_sent=float(pi),
                        q_perc=float(qi),
                        latency_ms=float(li),
                    )
                    for pi, qi, li in zip(p, q, latency)
                ]
            )
            np.testing.assert_array_equal(vector, scalar)

    def test_smooth_variant_has_no_hidden_deadline_cliff(self) -> None:
        spec = SPECS[1]
        below = spec.expected_utility(
            p_edge_admission_given_sent=1.0,
            q_perc=0.7,
            latency_ms=BUDGET_MS - 0.1,
        )
        above = spec.expected_utility(
            p_edge_admission_given_sent=1.0,
            q_perc=0.7,
            latency_ms=BUDGET_MS + 0.1,
        )
        self.assertTrue(math.isclose(below - above, 0.00025, abs_tol=1e-15))

    def test_larger_latency_quantile_lowers_reward_without_inferring_loss(self) -> None:
        spec = SPECS[2]
        low = spec.expected_utility(
            p_edge_admission_given_sent=0.8, q_perc=0.6, latency_ms=180.0
        )
        high = spec.expected_utility(
            p_edge_admission_given_sent=0.8, q_perc=0.6, latency_ms=260.0
        )
        self.assertGreater(low, high)
        # Non-admission mass is unchanged; the quantile changes only admitted utility.
        self.assertAlmostEqual(low - high, 0.8 * 0.25 * (80.0 / 200.0))

    def test_constrained_index_obeys_mask(self) -> None:
        reward = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
        latency = np.asarray([220.0, 190.0, 180.0], dtype=np.float64)
        index = _best_index(reward, latency, latency <= BUDGET_MS)
        self.assertEqual(index, 1)


if __name__ == "__main__":
    unittest.main()

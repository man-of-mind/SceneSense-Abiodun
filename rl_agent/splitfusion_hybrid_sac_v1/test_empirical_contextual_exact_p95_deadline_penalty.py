from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from .empirical_contextual_exact_p95_deadline_penalty import (
    BUDGET_MS,
    EXPECTED_TRAIN_CONTEXT_COUNT,
    ExactPenaltyDerivationError,
    REGISTERED_EXACT_PENALTY_SPEC_SHA256,
    _best_across_modes,
    base_p95_expected_utility,
    derive_exact_deadline_penalty,
    exact_penalty_spec_document,
    shaped_p95_expected_utility,
)
from .transaction_identity import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    PROJECT_ROOT
    / "experiments/splitfusion_hybrid_sac_fit_validation_v1"
    / "20260921_train_exact_p95_deadline_penalty_v1"
)


class ExactP95DeadlinePenaltyUnitTest(unittest.TestCase):
    def test_expected_utility_places_penalty_inside_admission(self) -> None:
        base = base_p95_expected_utility(
            p_admit=0.8, q_perc=0.7, latency_p95_ms=220.0
        )
        shaped = shaped_p95_expected_utility(
            p_admit=0.8,
            q_perc=0.7,
            latency_p95_ms=220.0,
            deadline_penalty=0.5,
        )
        self.assertEqual(shaped, base - 0.8 * 0.5)

    def test_exact_deadline_boundary_is_not_penalized(self) -> None:
        kwargs = dict(p_admit=0.9, q_perc=0.6, latency_p95_ms=BUDGET_MS)
        self.assertEqual(
            shaped_p95_expected_utility(**kwargs, deadline_penalty=100.0),
            base_p95_expected_utility(**kwargs),
        )

    def test_smooth_term_still_distinguishes_201_from_300(self) -> None:
        common = dict(p_admit=1.0, q_perc=0.7, deadline_penalty=2.0)
        at_201 = shaped_p95_expected_utility(latency_p95_ms=201.0, **common)
        at_300 = shaped_p95_expected_utility(latency_p95_ms=300.0, **common)
        self.assertGreater(at_201, at_300)
        self.assertAlmostEqual(at_201 - at_300, 0.25 * 99.0 / 200.0)

    def test_analytic_penalty_is_minimal_strict_float64(self) -> None:
        requirements = ((0.4, 0.5, 0.5), (0.3, 0.36, 0.6))
        delta, penalty, steps = derive_exact_deadline_penalty(requirements)
        self.assertAlmostEqual(delta, 0.2)
        self.assertGreater(penalty, delta)
        self.assertGreaterEqual(steps, 1)
        for feasible, infeasible, p_admit in requirements:
            self.assertLess(infeasible - p_admit * penalty, feasible)

    def test_positive_admission_exact_tie_has_finite_solution(self) -> None:
        delta, penalty, evaluations = derive_exact_deadline_penalty(
            ((0.4, 0.4, 0.5),)
        )
        self.assertEqual(delta, 0.0)
        self.assertTrue(math.isfinite(penalty))
        self.assertGreater(penalty, 0.0)
        self.assertLess(0.4 - 0.5 * penalty, 0.4)
        self.assertGreater(evaluations, 1)

    def test_zero_admission_is_excluded_from_derivation(self) -> None:
        delta, penalty, _ = derive_exact_deadline_penalty(((0.4, 100.0, 0.0),))
        self.assertEqual(delta, 0.0)
        self.assertEqual(penalty, math.nextafter(0.0, math.inf))
        delta, penalty, _ = derive_exact_deadline_penalty(((0.4, -1.0, 0.0),))
        self.assertEqual(delta, 0.0)
        self.assertEqual(penalty, math.nextafter(0.0, math.inf))

    def test_no_admitted_feasible_action_is_explicit(self) -> None:
        import numpy as np

        surfaces = [{
            "q": np.asarray([0], dtype=np.int64),
            "quality": np.asarray([0.8]),
            "payload": np.asarray([1000.0]),
            "datagrams": np.asarray([1], dtype=np.int64),
        }]
        networks = [{
            "p_admit": np.asarray([0.0]),
            "p50": np.asarray([100.0]),
            "p95": np.asarray([150.0]),
            "p99": np.asarray([180.0]),
        }]
        self.assertIsNone(
            _best_across_modes(surfaces, networks, feasible_only=True)
        )

    def test_shaped_tie_prefers_feasible_action_deterministically(self) -> None:
        import numpy as np

        surfaces = [{
            "q": np.asarray([0, 1], dtype=np.int64),
            "quality": np.asarray([0.5, 0.75]),
            "payload": np.asarray([1000.0, 1000.0]),
            "datagrams": np.asarray([1, 1], dtype=np.int64),
        }]
        networks = [{
            "p_admit": np.asarray([1.0, 1.0]),
            "p50": np.asarray([100.0, 100.0]),
            "p95": np.asarray([200.0, 300.0]),
            "p99": np.asarray([210.0, 310.0]),
        }]
        winner = _best_across_modes(
            surfaces, networks, feasible_only=False, penalty=0.125
        )
        self.assertIsNotNone(winner)
        self.assertEqual(winner.q_e4, 0)

    def test_specification_hash_is_pinned(self) -> None:
        self.assertEqual(
            canonical_sha256(exact_penalty_spec_document()),
            REGISTERED_EXACT_PENALTY_SPEC_SHA256,
        )

    def test_input_guards(self) -> None:
        with self.assertRaises(ValueError):
            base_p95_expected_utility(
                p_admit=1.1, q_perc=0.5, latency_p95_ms=100.0
            )
        with self.assertRaises(ValueError):
            shaped_p95_expected_utility(
                p_admit=1.0,
                q_perc=0.5,
                latency_p95_ms=100.0,
                deadline_penalty=-1.0,
            )


@unittest.skipUnless(EVIDENCE.exists(), "exact-penalty evidence not generated")
class ExactP95DeadlinePenaltyEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((EVIDENCE / "summary.json").read_text())
        cls.decision = json.loads(
            (EVIDENCE / "selection_decision.json").read_text()
        )

    def test_real_screen_coverage_and_exactness(self) -> None:
        exact = self.summary["exactness"]
        self.assertEqual(exact["train_context_count"], EXPECTED_TRAIN_CONTEXT_COUNT)
        self.assertEqual(
            exact["scalar_revalidated_winner_count"], EXPECTED_TRAIN_CONTEXT_COUNT
        )
        self.assertEqual(
            self.summary["decision"]["winner_identity_match_count"],
            EXPECTED_TRAIN_CONTEXT_COUNT,
        )

    def test_real_acceptance_is_fail_closed(self) -> None:
        expected = (
            "GO"
            if all(self.summary["decision"]["criteria"].values())
            else "NO_GO"
        )
        self.assertEqual(self.summary["decision"]["status"], expected)
        self.assertEqual(self.decision["status"], expected)

    def test_real_penalty_is_strictly_above_delta(self) -> None:
        decision = self.summary["decision"]
        self.assertGreater(decision["deadline_penalty"], decision["delta"])

    def test_scope_is_train_only_and_not_live(self) -> None:
        self.assertEqual(self.summary["scope"]["context_selection"], "TRAIN_IDS_ONLY")
        self.assertIn("LIVE_200_MS_SLA", self.summary["scope"]["claims_excluded"])
        self.assertEqual(
            self.summary["scope"]["fit_validation_scene_id_intersection_count"],
            0,
        )

    def test_registered_artifact_and_canonical_hashes_reconcile(self) -> None:
        for name, expected in self.summary["files"].items():
            self.assertEqual(hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(), expected)
        document = dict(self.summary)
        observed = document.pop("canonical_content_sha256")
        self.assertEqual(canonical_sha256(document), observed)

    def test_worst_requirement_reconciles_with_delta(self) -> None:
        worst = self.summary["worst_penalty_requirement"]
        self.assertEqual(
            worst["required_penalty_ratio"], self.summary["decision"]["delta"]
        )


if __name__ == "__main__":
    unittest.main()

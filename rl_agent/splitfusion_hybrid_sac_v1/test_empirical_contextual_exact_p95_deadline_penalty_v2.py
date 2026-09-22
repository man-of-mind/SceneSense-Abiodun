from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    EXPECTED_TRAIN_CONTEXT_COUNT,
    OLD_FLOAT64_V1_DEADLINE_PENALTY,
    REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
    base_p95_expected_utility64_v2,
    derive_float32_exact_deadline_penalty_v2,
    emitted_float32_target_v2,
    float32_exact_penalty_spec_document_v2,
    shaped_p95_expected_utility64_v2,
)
from .transaction_identity import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    PROJECT_ROOT
    / "experiments/splitfusion_hybrid_sac_fit_validation_v1"
    / "20260921_train_exact_p95_deadline_penalty_v2"
)


class ExactP95DeadlinePenaltyV2UnitTest(unittest.TestCase):
    def test_pinned_runtime_formula_and_float32_emission(self) -> None:
        p = 0.8
        quality = 0.7
        latency = 220.0
        expected = p * (quality - 0.25 * (latency / 200.0)) + (1.0 - p) * (-1.0)
        base64 = base_p95_expected_utility64_v2(
            p_admit=p, q_perc=quality, latency_p95_ms=latency
        )
        self.assertEqual(base64, expected)
        shaped64 = shaped_p95_expected_utility64_v2(
            p_admit=p,
            q_perc=quality,
            latency_p95_ms=latency,
            deadline_penalty=0.5,
        )
        self.assertEqual(shaped64, base64 - p * 0.5)
        import torch

        self.assertEqual(
            emitted_float32_target_v2(shaped64),
            torch.tensor(shaped64, dtype=torch.float32).item(),
        )

    def test_exact_deadline_is_not_penalized(self) -> None:
        base64 = base_p95_expected_utility64_v2(
            p_admit=0.9, q_perc=0.6, latency_p95_ms=200.0
        )
        shaped64 = shaped_p95_expected_utility64_v2(
            p_admit=0.9,
            q_perc=0.6,
            latency_p95_ms=200.0,
            deadline_penalty=100.0,
        )
        self.assertEqual(shaped64, base64)

    def test_zero_admission_emits_exact_minus_one(self) -> None:
        for quality, latency in ((0.0, 0.0), (1.0, 999.0)):
            base64 = base_p95_expected_utility64_v2(
                p_admit=0.0, q_perc=quality, latency_p95_ms=latency
            )
            self.assertEqual(base64, -1.0)
            self.assertEqual(emitted_float32_target_v2(base64), -1.0)

    def test_old_float64_strict_pair_collides_after_float32_cast(self) -> None:
        feasible64 = -0.13886236214769926
        old_infeasible_shaped64 = -0.13886236214769937
        self.assertLess(old_infeasible_shaped64, feasible64)
        self.assertEqual(OLD_FLOAT64_V1_DEADLINE_PENALTY, 0.5742957604173842)
        self.assertEqual(
            emitted_float32_target_v2(old_infeasible_shaped64),
            emitted_float32_target_v2(feasible64),
        )
        self.assertEqual(
            emitted_float32_target_v2(feasible64), -0.13886235654354095
        )

    def test_ordinal_search_proves_new_strict_order_and_predecessor_failure(self) -> None:
        target = emitted_float32_target_v2(-0.13886236214769926)
        p = 0.9919169403402207
        infeasible_base64 = 0.4307913313758727
        penalty, evaluations, predecessor, predecessor_insufficient = (
            derive_float32_exact_deadline_penalty_v2(
                ((target, infeasible_base64, p),)
            )
        )
        selected = emitted_float32_target_v2(infeasible_base64 - p * penalty)
        previous = emitted_float32_target_v2(
            infeasible_base64 - p * predecessor
        )
        self.assertLess(selected, target)
        self.assertGreaterEqual(previous, target)
        self.assertTrue(predecessor_insufficient)
        self.assertEqual(predecessor, math.nextafter(penalty, -math.inf))
        self.assertGreater(evaluations, 1)

    def test_derivation_rejects_zero_admission_requirements(self) -> None:
        with self.assertRaises(ValueError):
            derive_float32_exact_deadline_penalty_v2(((0.0, 1.0, 0.0),))

    def test_specification_hash_is_pinned(self) -> None:
        self.assertEqual(
            canonical_sha256(float32_exact_penalty_spec_document_v2()),
            REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
        )


@unittest.skipUnless(EVIDENCE.exists(), "v2 float32 exact-penalty evidence not generated")
class ExactP95DeadlinePenaltyV2EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((EVIDENCE / "summary_v2.json").read_text())
        cls.decision = json.loads(
            (EVIDENCE / "selection_decision_v2.json").read_text()
        )

    def test_two_pass_exhaustive_coverage_and_scalar_revalidation(self) -> None:
        exact = self.summary["exactness"]
        expected = EXPECTED_TRAIN_CONTEXT_COUNT * exact["actions_per_context"]
        self.assertEqual(exact["first_pass_action_context_evaluations"], expected)
        self.assertEqual(exact["second_pass_action_context_evaluations"], expected)
        self.assertEqual(
            exact["scalar_constrained_revalidated_winner_count"],
            EXPECTED_TRAIN_CONTEXT_COUNT,
        )
        self.assertEqual(
            exact["scalar_shaped_revalidated_winner_count"],
            EXPECTED_TRAIN_CONTEXT_COUNT,
        )

    def test_new_lambda_is_strict_and_predecessor_is_not(self) -> None:
        decision = self.summary["decision"]
        self.assertEqual(decision["strict_ordering_violation_count"], 0)
        self.assertTrue(decision["predecessor_insufficient"])
        self.assertGreater(decision["predecessor_violation_count"], 0)
        self.assertEqual(
            decision["predecessor"],
            math.nextafter(decision["deadline_penalty"], -math.inf),
        )

    def test_old_collision_is_reproduced_with_identical_bits(self) -> None:
        collision = self.summary["old_float64_v1_collision"]
        self.assertTrue(collision["float64_strict_ordering"])
        self.assertTrue(collision["emitted_float32_collision"])
        self.assertFalse(collision["emitted_float32_strict_ordering"])
        self.assertEqual(
            collision["constrained_emitted_float32_bits_hex"],
            collision["infeasible_emitted_float32_bits_hex"],
        )
        self.assertEqual(collision["context_index"], 938)
        self.assertEqual(collision["infeasible_mode_id"], 2)
        self.assertEqual(collision["infeasible_q_e4"], 9000)

    def test_winners_and_retention_are_go(self) -> None:
        decision = self.summary["decision"]
        constrained = self.summary["emitted_float32_constrained_oracle"]
        self.assertEqual(decision["status"], "GO")
        self.assertEqual(
            decision["winner_identity_match_count"], EXPECTED_TRAIN_CONTEXT_COUNT
        )
        self.assertEqual(constrained["p95_miss_count"], 0)
        self.assertGreaterEqual(constrained["quality_retention"], 0.95)
        self.assertGreaterEqual(constrained["admission_change"], -0.001)

    def test_scope_is_train_only(self) -> None:
        scope = self.summary["scope"]
        self.assertEqual(scope["context_selection"], "TRAIN_IDS_ONLY")
        self.assertEqual(scope["fit_validation_outcome_query_count"], 0)
        self.assertEqual(scope["fit_validation_scene_id_intersection_count"], 0)

    def test_evidence_hashes_reconcile(self) -> None:
        for name, expected in self.summary["files"].items():
            self.assertEqual(
                hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                expected,
            )
        document = dict(self.summary)
        observed = document.pop("canonical_content_sha256")
        self.assertEqual(canonical_sha256(document), observed)


if __name__ == "__main__":
    unittest.main()

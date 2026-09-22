"""Focused tests for the train-only P95 hinge coefficient screen."""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_p95_hinge_selection import (
    BUDGET_MS,
    EXPECTED_TRAIN_CONTEXT_COUNT,
    HINGE_LAMBDA_GRID,
    P95_HINGE_SPEC_SHA256,
    MEAN_ADMISSION_MAX_ABSOLUTE_DROP,
    MEAN_QUALITY_MINIMUM_RETENTION,
    P95DeadlineHingeSpecV1,
    _reward_vector,
    _suite_document,
    _training_contexts,
    select_smallest_eligible_lambda,
)
from .empirical_contextual_split_oracle import OracleAuditError
from .transaction_identity import canonical_sha256


class P95DeadlineHingeSpecTests(unittest.TestCase):
    def test_suite_hash_is_pinned(self) -> None:
        self.assertEqual(canonical_sha256(_suite_document()), P95_HINGE_SPEC_SHA256)

    def test_grid_and_criterion_are_small_and_fixed(self) -> None:
        self.assertEqual(
            HINGE_LAMBDA_GRID,
            (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        )
        self.assertEqual(MEAN_QUALITY_MINIMUM_RETENTION, 0.95)
        self.assertEqual(MEAN_ADMISSION_MAX_ABSOLUTE_DROP, 0.001)
        self.assertEqual(BUDGET_MS, 200.0)

    def test_hinge_is_zero_at_and_below_deadline(self) -> None:
        spec = P95DeadlineHingeSpecV1(4.0)
        for latency in (0.0, 199.0, 200.0):
            expected = 0.8 - 0.25 * latency / 200.0
            self.assertEqual(
                spec.expected_utility(
                    p_edge_admission_given_sent=1.0,
                    q_perc=0.8,
                    latency_p95_ms=latency,
                ),
                expected,
            )

    def test_hinge_penalizes_only_normalized_excess(self) -> None:
        spec = P95DeadlineHingeSpecV1(4.0)
        self.assertAlmostEqual(
            spec.expected_utility(
                p_edge_admission_given_sent=1.0,
                q_perc=0.8,
                latency_p95_ms=250.0,
            ),
            0.8 - 0.25 * 1.25 - 4.0 * 0.25,
            places=15,
        )

    def test_non_admission_semantics_are_preserved(self) -> None:
        spec = P95DeadlineHingeSpecV1(16.0)
        self.assertEqual(
            spec.expected_utility(
                p_edge_admission_given_sent=0.0,
                q_perc=1.0,
                latency_p95_ms=10_000.0,
            ),
            -1.0,
        )

    def test_vector_matches_scalar(self) -> None:
        spec = P95DeadlineHingeSpecV1(2.0)
        p_admit = np.asarray([1.0, 0.5, 0.0])
        quality = np.asarray([0.8, 0.6, 0.4])
        latency = np.asarray([180.0, 240.0, 300.0])
        vector = _reward_vector(spec, p_admit, quality, latency)
        scalar = np.asarray(
            [
                spec.expected_utility(
                    p_edge_admission_given_sent=float(p),
                    q_perc=float(q),
                    latency_p95_ms=float(value),
                )
                for p, q, value in zip(p_admit, quality, latency)
            ]
        )
        np.testing.assert_allclose(vector, scalar, rtol=0.0, atol=0.0)


class TrainOnlySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partition = load_registered_empirical_fit_partition()

    def test_contexts_are_train_only_and_complete(self) -> None:
        contexts = _training_contexts(self.partition)
        self.assertEqual(len(contexts), EXPECTED_TRAIN_CONTEXT_COUNT)
        training_ids = {
            row.sample_id
            for row in self.partition.scene_assignments
            if row.split == TRAIN_SPLIT
        }
        validation_ids = {
            row.sample_id
            for row in self.partition.scene_assignments
            if row.split == FIT_VALIDATION_SPLIT
        }
        observed = {row.sample_id for row in contexts}
        self.assertEqual(observed, training_ids)
        self.assertTrue(observed.isdisjoint(validation_ids))
        self.assertEqual(len(contexts), len(training_ids) * 4)

    def test_selector_chooses_smallest_eligible_coefficient(self) -> None:
        summaries = [
            {
                "hinge_lambda": value,
                "train_p95_budget_miss_count": 1 if value < 4.0 else 0,
                "mean_q_perc": 0.8 if value < 4.0 else 0.77,
                "mean_p_edge_admission_given_sent": 0.9,
            }
            for value in HINGE_LAMBDA_GRID
        ]
        self.assertEqual(
            float(select_smallest_eligible_lambda(summaries)["hinge_lambda"]),
            4.0,
        )

    def test_selector_refuses_no_eligible_coefficient(self) -> None:
        summaries = [
            {
                "hinge_lambda": value,
                "train_p95_budget_miss_count": 1,
                "mean_q_perc": 0.8,
                "mean_p_edge_admission_given_sent": 0.9,
            }
            for value in HINGE_LAMBDA_GRID
        ]
        self.assertIsNone(select_smallest_eligible_lambda(summaries))

    def test_selector_enforces_quality_and_admission_guards(self) -> None:
        summaries = []
        for value in HINGE_LAMBDA_GRID:
            summaries.append(
                {
                    "hinge_lambda": value,
                    "train_p95_budget_miss_count": 1 if value < 4.0 else 0,
                    "mean_q_perc": 0.8 if value == 0.0 else 0.75,
                    "mean_p_edge_admission_given_sent": (
                        0.9 if value == 0.0 else 0.898
                    ),
                }
            )
        self.assertIsNone(select_smallest_eligible_lambda(summaries))

    def test_selection_module_has_no_validation_panel_loader(self) -> None:
        from . import empirical_contextual_p95_hinge_selection as module

        source = inspect.getsource(module)
        self.assertNotIn("load_registered_fit_validation_panel", source)
        self.assertNotIn("FitValidationActorEvaluatorV1", source)
        self.assertNotIn("cuda(", source.lower())


if __name__ == "__main__":
    unittest.main()

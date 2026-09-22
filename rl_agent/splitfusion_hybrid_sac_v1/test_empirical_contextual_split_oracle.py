"""Focused tests for the exact frozen-panel SPLIT oracle analysis."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from . import empirical_contextual_split_oracle as oracle
from .empirical_contextual_baseline_runner import REGISTERED_BASELINE_CONFIG
from .modeled_smoke_support import MODELED_SMOKE_MODE_Q_E4_BOUNDS
from .offline_quality_grid.contract import Q_E4_GRID
from .payload_network_surrogate import build_payload_network_surrogate


class ExactActionDomainTest(unittest.TestCase):
    def test_every_inclusive_wire_action_is_enumerated_once(self) -> None:
        actions = oracle.enumerate_supported_actions()
        self.assertEqual(len(actions), 52_240)
        self.assertEqual(len(set(actions)), len(actions))
        offset = 0
        for mode_id, (lower, upper) in enumerate(MODELED_SMOKE_MODE_Q_E4_BOUNDS):
            width = upper - lower + 1
            self.assertEqual(actions[offset], (mode_id, lower))
            self.assertEqual(actions[offset + width - 1], (mode_id, upper))
            offset += width
        self.assertEqual(offset, len(actions))

    def test_registered_seed_random_is_deterministic_and_supported(self) -> None:
        first = [
            oracle.choose_registered_random_action(seed=seed, panel_index=index)
            for seed in REGISTERED_BASELINE_CONFIG.seeds
            for index in range(340)
        ]
        second = [
            oracle.choose_registered_random_action(seed=seed, panel_index=index)
            for seed in REGISTERED_BASELINE_CONFIG.seeds
            for index in range(340)
        ]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 900)
        for mode_id, q_e4 in first:
            lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
            self.assertLessEqual(lower, q_e4)
            self.assertLessEqual(q_e4, upper)

    def test_random_rejects_unregistered_seed_and_panel(self) -> None:
        with self.assertRaises(oracle.OracleAuditError):
            oracle.choose_registered_random_action(seed=999, panel_index=0)
        with self.assertRaises(oracle.OracleAuditError):
            oracle.choose_registered_random_action(seed=17, panel_index=340)


class VectorizedRegisteredCurveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = build_payload_network_surrogate()

    def test_vector_curve_is_scalar_exact_at_centers_and_interiors(self) -> None:
        for model in self.network.profile_models.values():
            curves = (model.reassembly_curve, model.admission_curve, *model.latency_curves.values())
            for curve in curves:
                centers = [block.x_center for block in curve.blocks]
                points = [curve.raw_x_min, curve.raw_x_max, *centers]
                points.extend((left + right) / 2.0 for left, right in zip(centers, centers[1:]))
                x = np.asarray(sorted(set(points)), dtype=np.float64)
                values, support = oracle._curve_vector(curve, x)
                for index, coordinate in enumerate(x):
                    scalar = curve.predict(float(coordinate))
                    self.assertAlmostEqual(values[index], scalar.value, places=13)
                    self.assertEqual(support[index], scalar.effective_support)

    def test_local_latency_support_matches_scalar_at_knots_and_interiors(self) -> None:
        for model in self.network.profile_models.values():
            knots = [pair[0] for pair in model.latency_support_knots]
            points = [*knots]
            points.extend((left + right) / 2.0 for left, right in zip(knots, knots[1:]))
            x = np.asarray(sorted(set(points)), dtype=np.float64)
            vector = oracle._local_latency_support_vector(model, x)
            expected = np.asarray([model.local_latency_support(float(value)) for value in x])
            np.testing.assert_array_equal(vector, expected)

    def test_curve_refuses_extrapolation(self) -> None:
        curve = next(iter(self.network.profile_models.values())).reassembly_curve
        with self.assertRaises(oracle.OracleAuditError):
            oracle._curve_vector(curve, np.asarray([curve.raw_x_min - 1e-6]))

    def test_latency_support_refuses_ambiguous_duplicate_knots(self) -> None:
        @dataclass
        class Fake:
            latency_support_knots: tuple

        model = Fake(((0.0, 100), (0.0, 110), (1.0, 120)))
        with self.assertRaises(oracle.OracleAuditError):
            oracle._local_latency_support_vector(model, np.asarray([0.5]))


class VectorizedSurfaceTest(unittest.TestCase):
    def test_every_integer_matches_adjacent_endpoint_lerp(self) -> None:
        class Row:
            def __init__(self, q_e4: int, payload: int, quality: float) -> None:
                self.q_e4 = q_e4
                self.total_transmitted_bytes = payload
                self._quality = quality

            def component(self, name: str) -> SimpleNamespace:
                self_test.assertEqual(name, "q_perc")
                return SimpleNamespace(valid=True, value=self._quality)

        class Surface:
            def __init__(self, rows: tuple[Row, ...]) -> None:
                self.rows = rows

            def _rows_for(self, sample_id: str, mode_id: int) -> tuple[Row, ...]:
                self_test.assertEqual(sample_id, "synthetic")
                self_test.assertEqual(mode_id, 11)
                return self.rows

        self_test = self
        rows = tuple(
            Row(q, 500_000 - 37 * q, (index * index + 3) / 130.0)
            for index, q in enumerate(Q_E4_GRID)
        )
        vector = oracle._surface_mode_vector(Surface(rows), "synthetic", 11)
        self.assertEqual(len(vector["q"]), 9792)
        for index, q in enumerate(vector["q"]):
            q_int = int(q)
            exact = next((row for row in rows if row.q_e4 == q_int), None)
            if exact is not None:
                expected_payload = float(exact.total_transmitted_bytes)
                expected_quality = exact._quality
            else:
                lower = max(
                    (row for row in rows if row.q_e4 < q_int),
                    key=lambda row: row.q_e4,
                )
                upper = min(
                    (row for row in rows if row.q_e4 > q_int),
                    key=lambda row: row.q_e4,
                )
                alpha = (q_int - lower.q_e4) / (upper.q_e4 - lower.q_e4)
                expected_payload = lower.total_transmitted_bytes + alpha * (
                    upper.total_transmitted_bytes - lower.total_transmitted_bytes
                )
                expected_quality = lower._quality + alpha * (
                    upper._quality - lower._quality
                )
            self.assertAlmostEqual(vector["payload"][index], expected_payload, places=10)
            self.assertAlmostEqual(vector["quality"][index], expected_quality, places=14)


class OracleSelectionTest(unittest.TestCase):
    def test_integer_interior_optimum_is_not_reduced_to_endpoints(self) -> None:
        objective = np.asarray([0.1, 0.3, 0.9, 0.4, 0.2])
        latency = np.asarray([100.0, 110.0, 120.0, 130.0, 140.0])
        self.assertEqual(oracle._best_index(objective, latency), 2)

    def test_constraint_is_inclusive_and_can_be_infeasible(self) -> None:
        objective = np.asarray([0.1, 0.5, 0.9])
        latency = np.asarray([201.0, 200.0, 250.0])
        self.assertEqual(
            oracle._best_index(objective, latency, latency <= 200.0), 1
        )
        self.assertIsNone(
            oracle._best_index(objective, latency, latency <= 199.0)
        )

    def test_tie_rule_prefers_lower_latency_then_lower_action(self) -> None:
        objective = np.asarray([0.8, 0.8, 0.8])
        latency = np.asarray([140.0, 130.0, 130.0])
        self.assertEqual(oracle._best_index(objective, latency), 1)

    def test_current_reward_is_not_binary_at_deadline(self) -> None:
        p = np.asarray([1.0, 1.0])
        quality = np.asarray([0.8, 0.8])
        rewards = oracle._reward_vector(
            p, quality, np.asarray([201.0, 300.0])
        )
        self.assertGreater(rewards[0], rewards[1])
        self.assertAlmostEqual(rewards[0] - rewards[1], 0.12375)


if __name__ == "__main__":
    unittest.main()

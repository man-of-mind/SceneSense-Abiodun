"""Focused tests for the presentation-only Hybrid-SAC results renderer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from .render_preliminary_results import (
        PROFILES,
        SEEDS,
        UPDATES,
        load_inputs,
        quantile,
        rolling_mean,
    )
except ImportError:  # Direct file execution.
    from render_preliminary_results import (  # type: ignore
        PROFILES,
        SEEDS,
        UPDATES,
        load_inputs,
        quantile,
        rolling_mean,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE = PROJECT_ROOT / "experiments/splitfusion_hybrid_sac_preliminary_baseline_v1/20260921_train_split_5000x3_v1"
VALIDATION = PROJECT_ROOT / "experiments/splitfusion_hybrid_sac_fit_validation_v1/20260921_three_seed_checkpoints_v1"


class RendererHelpersTest(unittest.TestCase):
    def test_trailing_rolling_mean_contains_only_complete_windows(self) -> None:
        actual = rolling_mean([1.0, 2.0, 3.0, 8.0], window=3)
        np.testing.assert_allclose(actual, [2.0, 13.0 / 3.0])

    def test_quantile_does_not_zero_exact_integer_positions(self) -> None:
        values = list(range(85))
        self.assertEqual(quantile(values, 0.25), 21.0)
        self.assertEqual(quantile(values, 0.50), 42.0)
        self.assertEqual(quantile(values, 0.75), 63.0)

    def test_real_inputs_have_registered_complete_shape(self) -> None:
        inputs = load_inputs(BASELINE, VALIDATION)
        self.assertEqual(set(inputs.training), set(SEEDS))
        self.assertTrue(all(len(rows) == 5000 for rows in inputs.training.values()))
        self.assertEqual(
            len(inputs.validation_aggregate), len(SEEDS) * len(UPDATES) * (len(PROFILES) + 1)
        )
        self.assertEqual(len(inputs.validation_context), len(SEEDS) * len(UPDATES) * 340)


if __name__ == "__main__":
    unittest.main()

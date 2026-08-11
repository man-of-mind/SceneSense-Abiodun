import unittest

import pandas as pd

from data_collection.analyze_detection_ab_gate import (
    _longest_true_dwell,
    _paired_block_bootstrap_lift,
    _trajectory_comparison,
)


class DetectionABGateTests(unittest.TestCase):
    def test_longest_true_dwell_respects_gaps(self):
        dwell = _longest_true_dwell(
            [True, True, True, True, False, True, True],
            [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3],
        )
        self.assertAlmostEqual(dwell, 0.3)

    def test_matched_trajectory_passes_small_deltas(self):
        baseline = pd.DataFrame(
            {
                "in_scope": [True, True, False],
                "distance_m": [20.0, 20.1, 26.0],
                "projected_x": [400.0, 401.0, 900.0],
                "projected_y": [200.0, 201.0, 200.0],
            }
        )
        candidate = baseline.copy()
        candidate["distance_m"] += 0.1
        candidate["projected_x"] += 1.0
        result = _trajectory_comparison(
            baseline,
            candidate,
            {
                "maximum_target_row_delta_fraction": 0.1,
                "minimum_in_scope_sequence_agreement": 0.9,
                "maximum_median_distance_delta_m": 0.75,
                "maximum_median_projection_delta_px": 10.0,
            },
        )
        self.assertTrue(result["pair_valid"])

    def test_matched_trajectory_rejects_scope_mismatch(self):
        baseline = pd.DataFrame(
            {
                "in_scope": [True] * 10,
                "distance_m": [20.0] * 10,
                "projected_x": [400.0] * 10,
                "projected_y": [200.0] * 10,
            }
        )
        candidate = baseline.copy()
        candidate["in_scope"] = [False] * 10
        result = _trajectory_comparison(
            baseline,
            candidate,
            {
                "maximum_target_row_delta_fraction": 0.1,
                "minimum_in_scope_sequence_agreement": 0.9,
                "maximum_median_distance_delta_m": 0.75,
                "maximum_median_projection_delta_px": 10.0,
            },
        )
        self.assertFalse(result["pair_valid"])

    def test_paired_bootstrap_reports_positive_lift(self):
        baseline = [False] * 80
        candidate = [False] * 40 + [True] * 40
        result = _paired_block_bootstrap_lift(
            baseline,
            candidate,
            replicates=2000,
            block_length=5,
            seed=7,
        )
        self.assertEqual(result["paired_rows"], 80)
        self.assertAlmostEqual(result["lift_pp"], 50.0)
        self.assertGreater(result["ci95_lower_pp"], 0.0)

    def test_paired_bootstrap_rejects_unpaired_arrays(self):
        with self.assertRaises(ValueError):
            _paired_block_bootstrap_lift(
                [False, True],
                [True],
                replicates=100,
                block_length=2,
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()

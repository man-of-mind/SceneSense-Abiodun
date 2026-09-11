from __future__ import annotations

import unittest

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288_final_v3 as candidate,
)


class FinalV3CalibrationTest(unittest.TestCase):
    def test_family_calibration_is_positive_and_provenanced(self) -> None:
        calibration, publication = candidate._final_calibration()
        self.assertEqual(set(calibration), {"noAE", "AE128", "AE64", "AE32"})
        self.assertEqual(
            calibration["noAE"]["target_variant"],
            "SYNCHRONIZATION_LIGHT_V2",
        )
        for family in ("AE128", "AE64", "AE32"):
            self.assertEqual(
                calibration[family]["target_variant"],
                "OVERLAPPED_OUTPUT_PRESERVING_V3",
            )
        for family, row in calibration.items():
            self.assertGreater(row["total_edge_processing_reduction_ms"], 0.0)
            self.assertGreater(row["optimized_total_edge_processing_ms_median"], 0.0)
            self.assertGreater(len(publication[family]), 0)

    def test_action15_failure_is_explicitly_nonfinite(self) -> None:
        failure = candidate.source._load_json(candidate.ACTION15_FAILURE)
        self.assertIn("non-finite", failure["failure"])


if __name__ == "__main__":
    unittest.main()

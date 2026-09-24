#!/usr/bin/env python3
"""Offline tests for the analysis path. CPU only, synthetic inputs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rl_agent.ue_mcs_backlog_calibration_v1 import analyze as A  # noqa: E402
from rl_agent.ue_mcs_backlog_calibration_v1 import contract as C  # noqa: E402


def row(**over):
    base = {
        "cell_id": "c0", "profile_id": "ADVERSE_STABLE", "tier": "low",
        "block_index": 0, "since_transition": 0, "decision_index": 0,
        "repetition": "0", "previous_tier": "", "mcs": 10.0, "backlog": 100.0,
        "latency_ms": 5.0, "complete": True, "in_budget": True,
        "mcs_status": "OBSERVED", "backlog_status": "OBSERVED",
        "mcs_age_ms": "10.0", "backlog_age_ms": "1.0",
        "chunks_per_frame": "1", "chunks_sent": "1", "chunks_dropped": "0",
        "unique_chunks_received": "1", "terminal_outcome": "COMPLETE_REASSEMBLY",
    }
    base.update(over)
    return base


class WindowTests(unittest.TestCase):
    def test_steady_window_excludes_the_transient(self):
        cutoff = C.FRAMES_PER_BLOCK - C.STEADY_STATE_DECISIONS
        rows = [row(since_transition=i) for i in range(C.FRAMES_PER_BLOCK)]
        steady = A.steady_rows(rows)
        self.assertTrue(all(r["since_transition"] >= cutoff for r in steady))
        self.assertTrue(all(r["since_transition"] >= C.TRANSIENT_DECISIONS
                            for r in steady))

    def test_transient_ignores_the_first_block(self):
        rows = [row(block_index=0, since_transition=1),
                row(block_index=1, since_transition=1, previous_tier="low",
                    tier="high")]
        out = A.transient_analysis(rows)
        self.assertIn("ADVERSE_STABLE", out)
        self.assertIn("low->high", out["ADVERSE_STABLE"])
        self.assertEqual(len(out["ADVERSE_STABLE"]), 1)


class SaturationTests(unittest.TestCase):
    def test_ceiling_fraction_is_reported(self):
        rows = ([row(backlog=1000.0) for _ in range(9)]
                + [row(backlog=10.0)])
        out = A.saturation_analysis(rows)
        self.assertEqual(out["observed_ceiling_bytes"], 1000.0)
        self.assertAlmostEqual(out["overall_at_ceiling_fraction"], 0.9)

    def test_empty_input_is_reported_not_crashed(self):
        self.assertEqual(A.saturation_analysis([])["decisions"], 0)


class NextFrameTests(unittest.TestCase):
    def test_target_comes_from_the_following_decision(self):
        rows = [row(decision_index=0, complete=True),
                row(decision_index=1, complete=False)]
        out = A.next_frame_targets(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["y_complete"], 0.0)

    def test_gaps_in_the_decision_index_are_not_bridged(self):
        rows = [row(decision_index=0), row(decision_index=5)]
        self.assertEqual(A.next_frame_targets(rows), [])

    def test_block_boundary_crossings_are_flagged(self):
        rows = [row(decision_index=0, block_index=0),
                row(decision_index=1, block_index=1)]
        out = A.next_frame_targets(rows)
        self.assertTrue(out[0]["crosses_block_boundary"])


class MetricTests(unittest.TestCase):
    def test_auc_is_one_for_a_perfect_ranking(self):
        self.assertEqual(A.roc_auc([0.1, 0.2, 0.9], [0.0, 0.0, 1.0]), 1.0)

    def test_auc_is_half_for_ties(self):
        self.assertEqual(A.roc_auc([0.5, 0.5], [0.0, 1.0]), 0.5)

    def test_auc_is_missing_for_a_single_class(self):
        self.assertIsNone(A.roc_auc([0.1, 0.9], [1.0, 1.0]))

    def test_cliffs_delta_is_one_for_full_separation(self):
        out = A.cliffs_delta([5.0] * 10, [1.0] * 10)
        self.assertEqual(out["delta"], 1.0)
        self.assertEqual(out["interpretation"], "LARGE")


class BlockedPredictionTests(unittest.TestCase):
    def test_folds_are_whole_cells(self):
        samples = []
        for cell in range(4):
            for index in range(40):
                backlog = float(index % 20)
                samples.append({
                    "cell_id": f"c{cell}", "profile_id": "ADVERSE_STABLE",
                    "block_index": 0, "tier": "low", "mcs": 10.0,
                    "backlog": backlog,
                    "crosses_block_boundary": False,
                    "y_complete": 1.0 if backlog < 10 else 0.0,
                    "y_latency_ms": 5.0, "y_in_budget": 1.0})
        out = A.blocked_prediction(samples, "y_complete")
        self.assertEqual(out["B_backlog_only"]["folds_scored"], 4)
        # Backlog fully determines the label here, MCS is constant, so backlog
        # must beat MCS-only, which cannot separate at all.
        self.assertGreater(out["B_backlog_only"]["auc_mean"],
                           out["A_mcs_only"]["auc_mean"])
        self.assertEqual(out["comparison"]["split"], "LEAVE_ONE_CELL_OUT_BLOCKED")

    def test_incomplete_case_rows_are_dropped_not_imputed(self):
        samples = [{"cell_id": "c0", "profile_id": "A", "block_index": 0,
                    "tier": "low", "mcs": None, "backlog": 1.0,
                    "crosses_block_boundary": False, "y_complete": 1.0,
                    "y_latency_ms": None, "y_in_budget": None}] * 10
        out = A.blocked_prediction(samples, "y_complete")
        self.assertEqual(out["usable"], 0)
        self.assertEqual(out["dropped"], 10)


class MissingnessTests(unittest.TestCase):
    def test_coverage_counts_only_observed(self):
        rows = [row(mcs_status="OBSERVED"), row(mcs_status="MISSING_STALE"),
                row(mcs_status="MISSING_NO_PRIOR_GRANT")]
        out = A.missingness_analysis(rows)
        entry = out["per_cell"]["c0"]
        self.assertAlmostEqual(entry["mcs_coverage"], 1 / 3)
        self.assertEqual(entry["mcs_missing_stale"], 1)
        self.assertEqual(entry["mcs_missing_no_prior"], 1)


class NormalizationTests(unittest.TestCase):
    def test_bounds_warn_against_the_deployed_scale(self):
        out = A.normalization_bounds([row(backlog=float(v)) for v in range(1, 101)])
        self.assertIn("log1p_scale=1.0",
                      out["pre_enqueue_backlog_bytes"]["warning"])
        self.assertIn("never encoded as 0",
                      out["previous_ul_mcs"]["missing_policy"])


if __name__ == "__main__":
    unittest.main()

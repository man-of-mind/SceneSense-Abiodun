from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from phase2_map_sharing.analyze_decision_opportunity_pilot_v1 import (
    DEFAULT_CONFIG,
    _augment_warning_evidence,
    _evaluate_gates,
    _validate_output,
    load_config,
)


class DecisionOpportunityPilotAnalysisTests(unittest.TestCase):
    def test_checked_in_config_is_exact_and_offline_only(self) -> None:
        config, warning_config, replay_config = load_config(DEFAULT_CONFIG)

        self.assertEqual(
            config["authorization"],
            "offline_immutable_pilot_decision_only_no_downstream",
        )
        self.assertEqual(
            config["frozen_dependencies"]["setting_id"],
            "c20_a30_t05_u00",
        )
        self.assertEqual(
            warning_config["source_tracker"]["algorithm"],
            "source_local_confirmed_cv.v3",
        )
        self.assertEqual(replay_config["truth_evaluation"]["cadence_s"], 0.1)

    def test_output_is_create_only_sibling_with_safe_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            batch = parent / "capture"
            batch.mkdir()
            valid = parent / "20260819_020611_decision"
            _validate_output(batch, valid)

            with self.assertRaisesRegex(ValueError, "capture sibling"):
                _validate_output(batch, batch / "inside_decision")
            with self.assertRaisesRegex(ValueError, "safe"):
                _validate_output(batch, parent / "unsafe")
            valid.mkdir()
            with self.assertRaises(FileExistsError):
                _validate_output(batch, valid)

    def test_target_attribution_uses_only_prior_or_current_track_match(self) -> None:
        target_tracks = pd.DataFrame(
            [
                {
                    "source_role": "helper",
                    "frame_id": 10,
                    "source_track_id": "helper:track:000001",
                }
            ]
        )
        base = {
            "evidence_sources": '["helper"]',
            "evidence_track_ids": '["helper:track:000001"]',
        }
        rows = _augment_warning_evidence(
            [
                {**base, "frame_id": 12},
                {**base, "frame_id": 8},
                {
                    "frame_id": 12,
                    "evidence_sources": '["recipient"]',
                    "evidence_track_ids": '["recipient:track:000001"]',
                },
            ],
            target_tracks,
        )

        self.assertEqual(rows[0]["helper_target_track_active_evidence"], 1)
        self.assertEqual(rows[0]["helper_target_last_matched_frame_id"], 10)
        self.assertEqual(rows[1]["helper_target_track_active_evidence"], 0)
        self.assertIsNone(rows[1]["helper_target_last_matched_frame_id"])
        self.assertEqual(rows[2]["helper_active_evidence"], 0)

    def test_nuisance_gate_fails_without_changing_positive_gates(self) -> None:
        config, _, _ = load_config(DEFAULT_CONFIG)
        confirmations = {
            "helper": {"carla_timestamp": 10.0},
            "recipient": {"carla_timestamp": 12.4},
        }
        warning_evidence = {
            arm: {
                "helper_warning": {"frame_id": 100},
                "lead_vs_ego_s": 3.3,
                "recipient_speed_mps": 4.15,
                "before_first_hidden_actor_yield": True,
            }
            for arm in ("send_everything", "hazard_only")
        }
        results = _evaluate_gates(
            config=config,
            helper_runs=[{"frame_count": 10}],
            confirmations=confirmations,
            warning_evidence=warning_evidence,
            target_misses={"ego_only": 0, "send_everything": 0, "hazard_only": 0},
            benign_rates={
                "ego_only": 3 / 70,
                "send_everything": 9 / 70,
                "hazard_only": 11 / 70,
            },
        )

        self.assertTrue(results["five_consecutive_helper_target_detections"])
        self.assertTrue(results["recipient_v3_confirmation_at_least_1s_later"])
        self.assertTrue(results["helper_warning_lead_at_least_0p5s_both_cooperative_arms"])
        self.assertFalse(
            results["benign_false_warning_active_rate_at_most_10pct_all_arms"]
        )
        self.assertFalse(results["cooperative_benign_excess_at_most_2pp"])


if __name__ == "__main__":
    unittest.main()

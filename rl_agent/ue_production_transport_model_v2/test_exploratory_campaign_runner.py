#!/usr/bin/env python3
"""Focused tests for the Run-4 exploratory campaign evidence gate."""

from __future__ import annotations

import copy
import unittest

from rl_agent.ue_production_transport_model_v2 import (
    exploratory_campaign_runner as campaign,
)


class HistoricalObservableGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = {
            "decision_count": 288,
            "success_count": 259,
            "failure_count": 29,
            "schedule_id": "schedule",
            "feature_diagnostics": [
                {"feature_name": "camera_si", "span": 1.0}
            ],
        }
        self.current_milestones = [
            {
                "update": update,
                "decision_count": 288 + 4 * update,
                "checkpoint_sha256": f"new-{update}",
                "metrics": (
                    None if update == 0 else {"update_index": update}
                ),
            }
            for update in (0, 100, 250, 500)
        ]
        self.historical_smoke = {
            "summary": {
                "final_checkpoint_sha256": (
                    campaign.HISTORICAL_SEED17_UPDATE500_SHA256
                )
            },
            "milestones": [
                {
                    **row,
                    "checkpoint_sha256": f"historical-{row['update']}",
                }
                for row in self.current_milestones
            ],
            "trajectory": {"decisions": 2288, "success_rate": 0.832},
        }

    def compare(self):
        return campaign._compare_historical_observables(
            historical_smoke=self.historical_smoke,
            historical_preflight=self.preflight,
            current_preflight=self.preflight,
            current_milestones=self.current_milestones,
            current_trajectory=self.historical_smoke["trajectory"],
        )

    def test_checkpoint_identity_is_not_compared_across_bindings(self):
        result = self.compare()
        self.assertEqual(
            result["status"], "HISTORICAL_OBSERVABLES_REPRODUCED"
        )
        self.assertFalse(
            result[
                "historical_checkpoint_identity_comparable_to_full_checkpoint"
            ]
        )

    def test_metric_drift_is_rejected(self):
        changed = copy.deepcopy(self.current_milestones)
        changed[-1]["metrics"]["update_index"] = 499
        with self.assertRaisesRegex(RuntimeError, "milestone metrics"):
            campaign._compare_historical_observables(
                historical_smoke=self.historical_smoke,
                historical_preflight=self.preflight,
                current_preflight=self.preflight,
                current_milestones=changed,
                current_trajectory=self.historical_smoke["trajectory"],
            )

    def test_preflight_drift_is_rejected(self):
        changed = copy.deepcopy(self.preflight)
        changed["failure_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "preflight"):
            campaign._compare_historical_observables(
                historical_smoke=self.historical_smoke,
                historical_preflight=self.preflight,
                current_preflight=changed,
                current_milestones=self.current_milestones,
                current_trajectory=self.historical_smoke["trajectory"],
            )

    def test_trajectory_drift_is_rejected(self):
        changed = copy.deepcopy(self.historical_smoke["trajectory"])
        changed["success_rate"] = 0.831
        with self.assertRaisesRegex(RuntimeError, "trajectory"):
            campaign._compare_historical_observables(
                historical_smoke=self.historical_smoke,
                historical_preflight=self.preflight,
                current_preflight=self.preflight,
                current_milestones=self.current_milestones,
                current_trajectory=changed,
            )


if __name__ == "__main__":
    unittest.main()

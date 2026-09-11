from __future__ import annotations

import unittest

from rl_agent.splitfusion_edge_freshness_scheduler_v1.counterfactual_288 import (
    _candidate_frames,
    _measured_install_summary,
    _systematic_select,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageReason,
)


class Counterfactual288Test(unittest.TestCase):
    def test_systematic_selection_is_exact_and_endpoint_inclusive(self) -> None:
        self.assertEqual(_systematic_select(list(range(10)), 0), set())
        self.assertEqual(_systematic_select(list(range(10)), 10), set(range(10)))
        selected = _systematic_select(list(range(10)), 4)
        self.assertEqual(len(selected), 4)
        self.assertIn(0, selected)
        self.assertIn(9, selected)

    def test_candidate_preserves_upstream_counts(self) -> None:
        cell = {
            "cell_id": "a01__favorable_stable",
            "edge_complete_reassemblies": "3",
            "edge_admissions": "2",
        }
        rows = []
        for index in range(4):
            capture = 1.0 + index * 0.1
            rows.append(
                {
                    "frame_id": str(100 + index),
                    "capture_wall_s": str(capture),
                    "edge_receipt_wall_s": "1.05" if index == 0 else "",
                    "edge_tail_complete_wall_s": "1.15" if index == 0 else "",
                    "map_installed_at": "1.16" if index == 0 else "",
                    "edge_timing_ns": (
                        "{'total_edge_processing': 100000000}"
                        if index == 0
                        else ""
                    ),
                    "payload_bytes": "1000",
                }
            )
        frames, counters = _candidate_frames(
            cell=cell,
            rows=rows,
            family_calibration={"total_edge_processing_reduction_ms": 40.0},
            publication_samples=[3_000_000],
            action_service_pool=[100_000_000],
            family_service_pool=[110_000_000],
            profile_delay_pool=[10_000_000],
            action_profile_arrival_pool=[50_000_000],
            profile_arrival_pool=[60_000_000],
        )
        self.assertEqual(sum(item.arrival_ns is not None for item in frames), 2)
        self.assertEqual(
            sum(
                item.pre_scheduler_reason
                is TwoStageReason.MEASURED_PRE_QUEUE_REJECTION
                for item in frames
            ),
            1,
        )
        self.assertEqual(
            sum(
                item.pre_scheduler_reason is TwoStageReason.TRANSPORT_INCOMPLETE
                for item in frames
            ),
            1,
        )
        self.assertEqual(counters["arrival_observed"], 1)
        self.assertEqual(counters["arrival_imputed_within_cell"], 1)
        self.assertEqual(
            counters["service_observed"]
            + counters["service_imputed_within_cell"]
            + counters["service_imputed_same_action"]
            + counters["service_imputed_same_family"],
            2,
        )

    def test_measured_install_summary_uses_only_newer_maps(self) -> None:
        rows = [
            {"capture_wall_s": "1.0", "map_installed_at": "1.2"},
            {"capture_wall_s": "1.1", "map_installed_at": "1.4"},
            {"capture_wall_s": "1.05", "map_installed_at": "1.5"},
        ]
        result = _measured_install_summary(rows)
        self.assertEqual(result["installed"], 3)
        self.assertEqual(result["useful_newer_map_installations"], 2)
        self.assertEqual(result["installed_within_100ms"], 0)
        self.assertEqual(result["installed_within_500ms"], 3)


if __name__ == "__main__":
    unittest.main()

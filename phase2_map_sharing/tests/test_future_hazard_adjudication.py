from __future__ import annotations

import unittest

import pandas as pd

from phase2_map_sharing.adjudicate_future_hazards import (
    _aligned_counterfactual_ego,
    _first_sustained_stop,
    _future_label,
    _safe_output_name,
    match_warnings_one_to_one,
    oriented_box_clearance_m,
)


class OneToOneMatchingTests(unittest.TestCase):
    def test_matching_maximizes_cardinality_before_distance(self) -> None:
        warnings = pd.DataFrame(
            [
                {"class_name": "vehicle", "track_world_x": 0.0, "track_world_y": 0.0},
                {"class_name": "vehicle", "track_world_x": 1.0, "track_world_y": 0.0},
            ],
            index=[10, 11],
        )
        truth = pd.DataFrame(
            [
                {
                    "class_name": "vehicle",
                    "actor_id": "near_both",
                    "role_name": "actor-a",
                    "origin_x": 0.9,
                    "origin_y": 0.0,
                },
                {
                    "class_name": "vehicle",
                    "actor_id": "only_second",
                    "role_name": "actor-b",
                    "origin_x": 1.9,
                    "origin_y": 0.0,
                },
            ]
        )
        matches = match_warnings_one_to_one(warnings, truth, gate_m=1.0)
        self.assertEqual(matches[10]["current_truth_actor_id"], "near_both")
        self.assertEqual(matches[11]["current_truth_actor_id"], "only_second")

    def test_truth_actor_is_never_reused_within_frame_and_class(self) -> None:
        warnings = pd.DataFrame(
            [
                {"class_name": "person", "track_world_x": 0.0, "track_world_y": 0.0},
                {"class_name": "pedestrian", "track_world_x": 0.1, "track_world_y": 0.0},
            ]
        )
        truth = pd.DataFrame(
            [
                {
                    "class_name": "pedestrian",
                    "actor_id": "walker",
                    "role_name": "target",
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                }
            ]
        )
        matches = match_warnings_one_to_one(warnings, truth, gate_m=5.0)
        matched = [
            value["current_truth_actor_id"]
            for value in matches.values()
            if value["current_truth_matched"]
        ]
        self.assertEqual(matched, ["walker"])


class FutureLabelTests(unittest.TestCase):
    @staticmethod
    def actor_truth(timestamps: list[float], distances: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame_id": list(range(len(timestamps))),
                "carla_timestamp": timestamps,
                "origin_x": distances,
                "origin_y": [0.0] * len(timestamps),
                "yaw_deg": [0.0] * len(timestamps),
                "length_m": [0.4] * len(timestamps),
                "width_m": [0.4] * len(timestamps),
            }
        )

    @staticmethod
    def ego(count: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame_id": list(range(count)),
                "recipient_x": [0.0] * count,
                "recipient_y": [0.0] * count,
                "recipient_yaw_deg": [0.0] * count,
                "recipient_speed_mps": [0.0] * count,
            }
        )

    def test_observed_hazard_is_positive_even_if_remaining_horizon_is_censored(self) -> None:
        truth = self.actor_truth([0.0, 0.1], [4.0, 2.0])
        result = _future_label(
            {"warning_at_s": 0.0},
            truth,
            self.ego(2),
            horizon_s=5.0,
            safety_radius_m=2.5,
            cadence_s=0.1,
            ego_dimensions=None,
        )
        self.assertEqual(result["future_label"], "truth_hazard_positive")
        self.assertEqual(result["false_warning"], 0)

    def test_complete_safe_horizon_is_negative_and_short_safe_horizon_is_censored(self) -> None:
        complete_times = [index / 10.0 for index in range(51)]
        complete = _future_label(
            {"warning_at_s": 0.0},
            self.actor_truth(complete_times, [10.0] * len(complete_times)),
            self.ego(len(complete_times)),
            horizon_s=5.0,
            safety_radius_m=2.5,
            cadence_s=0.1,
            ego_dimensions=None,
        )
        self.assertEqual(complete["future_label"], "truth_hazard_negative")
        self.assertEqual(complete["false_warning"], 1)

        short_times = [index / 10.0 for index in range(11)]
        short = _future_label(
            {"warning_at_s": 0.0},
            self.actor_truth(short_times, [10.0] * len(short_times)),
            self.ego(len(short_times)),
            horizon_s=5.0,
            safety_radius_m=2.5,
            cadence_s=0.1,
            ego_dimensions=None,
        )
        self.assertEqual(short["future_label"], "censored_before_full_horizon")
        self.assertIsNone(short["false_warning"])


class CounterfactualAlignmentTests(unittest.TestCase):
    @staticmethod
    def trace(
        frame_ids: list[int], elapsed: list[float], x_values: list[float]
    ) -> pd.DataFrame:
        count = len(frame_ids)
        return pd.DataFrame(
            {
                "frame_id": frame_ids,
                "elapsed_s": elapsed,
                "recipient_x": x_values,
                "recipient_y": [0.0] * count,
                "recipient_yaw_deg": [0.0] * count,
                "recipient_speed_mps": [1.0] * count,
            }
        )

    def test_maps_donor_motion_to_reference_frame_ids(self) -> None:
        reference = self.trace([100, 101], [0.1, 0.2], [10.0, 11.0])
        donor = self.trace([900, 901], [0.1, 0.2], [20.0, 21.0])
        aligned = _aligned_counterfactual_ego(reference, donor, cadence_s=0.1)
        self.assertEqual(aligned["frame_id"].tolist(), [100, 101])
        self.assertEqual(
            aligned["counterfactual_source_frame_id"].tolist(), [900, 901]
        )
        self.assertEqual(aligned["recipient_x"].tolist(), [20.0, 21.0])

    def test_rejects_pair_beyond_half_cadence(self) -> None:
        reference = self.trace([100], [0.1], [10.0])
        donor = self.trace([900], [0.16], [20.0])
        with self.assertRaisesRegex(ValueError, "half-cadence"):
            _aligned_counterfactual_ego(reference, donor, cadence_s=0.1)


class PhysicalOutcomeTests(unittest.TestCase):
    def test_oriented_box_clearance_handles_longitudinal_lateral_and_overlap(self) -> None:
        first = (0.0, 0.0, 0.0, 4.0, 2.0)
        self.assertAlmostEqual(
            oriented_box_clearance_m(first, (5.0, 0.0, 0.0, 4.0, 2.0)),
            1.0,
        )
        self.assertAlmostEqual(
            oriented_box_clearance_m(first, (0.0, 5.0, 0.0, 4.0, 2.0)),
            3.0,
        )
        self.assertEqual(
            oriented_box_clearance_m(first, (3.0, 0.0, 0.0, 4.0, 2.0)),
            0.0,
        )

    def test_sustained_stop_requires_consecutive_dwell_after_hazard_motion(self) -> None:
        trace = pd.DataFrame(
            {
                "frame_id": list(range(10)),
                "recipient_speed_mps": [2.0, 0.0, 0.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0],
            }
        )
        stop = _first_sustained_stop(
            trace,
            not_before_frame=2,
            speed_threshold_mps=0.2,
            dwell_s=0.5,
            cadence_s=0.1,
        )
        self.assertIsNotNone(stop)
        self.assertEqual(int(stop["frame_id"]), 4)

    def test_output_namespace_is_create_only_and_path_safe(self) -> None:
        self.assertEqual(
            _safe_output_name("hazard_adjudication_v1"),
            "hazard_adjudication_v1",
        )
        for value in ("../hazard_adjudication_v1", "/tmp/hazard_adjudication_v1", "evaluation_v4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _safe_output_name(value)


if __name__ == "__main__":
    unittest.main()

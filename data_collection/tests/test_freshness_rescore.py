import math
import unittest

import pandas as pd

from data_collection.rescore_policy_corpus_freshness import (
    _km_quantile,
    _quantile,
    dwell_segments,
    score_seeded_view,
    summarize_freshness,
)


def _objects(actor_id=1, speed_mps=10.0, observed_from=0.0, timestamps=None):
    timestamps = timestamps or [0.0, 0.05, 0.10, 0.15, 0.20]
    return pd.DataFrame(
        [
            {
                "episode_id": "episode_1",
                "scenario_family": "synthetic",
                "scenario_variant": "synthetic_a",
                "split": "train",
                "step_index": index,
                "timestamp_s": timestamp,
                "actor_id": actor_id,
                "class_name": "vehicle",
                "speed_mps": speed_mps,
                "observed": timestamp >= observed_from,
            }
            for index, timestamp in enumerate(timestamps)
        ]
    )


class FreshnessScoringTests(unittest.TestCase):
    def test_gt_seed_uses_locked_error_and_separates_near_from_breached(self):
        scored, tracks = score_seeded_view(
            _objects(), "gt_seeded_motion_only", 2.0, 1.11, 20.0, [3, 5, 10]
        )

        self.assertTrue(scored["mapped"].all())
        self.assertFalse(bool(scored.iloc[0]["near_breach_3tick"]))
        self.assertTrue(bool(scored.iloc[1]["near_breach_3tick"]))
        self.assertTrue(bool(scored.iloc[-1]["mapped_over_epsilon"]))
        self.assertFalse(bool(scored.iloc[-1]["near_breach_3tick"]))
        self.assertAlmostEqual(
            float(scored.iloc[-1]["localization_error_m"]),
            math.hypot(1.11, 10.0 * 0.20),
        )
        self.assertTrue(bool(tracks.iloc[0]["breached"]))
        self.assertAlmostEqual(float(tracks.iloc[0]["time_to_first_sampled_breach_s"]), 0.20)

    def test_detection_seed_marks_pre_seed_truth_as_strictly_unsafe(self):
        scored, _tracks = score_seeded_view(
            _objects(observed_from=0.10),
            "detection_seeded_deployable",
            2.0,
            1.11,
            20.0,
            [3, 5, 10],
        )

        self.assertEqual(int((~scored["mapped"]).sum()), 2)
        self.assertTrue(scored.iloc[:2]["strict_gt_unsafe"].all())
        self.assertFalse(scored.iloc[:2]["mapped_over_epsilon"].any())
        self.assertAlmostEqual(float(scored.iloc[2]["aoi_s"]), 0.0)

    def test_nonbreaching_track_is_right_censored(self):
        _scored, tracks = score_seeded_view(
            _objects(speed_mps=1.0, timestamps=[0.0, 0.05, 0.10]),
            "gt_seeded_motion_only",
            2.0,
            1.11,
            20.0,
            [3],
        )

        self.assertFalse(bool(tracks.iloc[0]["breached"]))
        self.assertTrue(bool(tracks.iloc[0]["right_censored"]))
        self.assertAlmostEqual(float(tracks.iloc[0]["censor_time_s"]), 0.10)

    def test_all_object_liveness_deduplicates_shared_frames(self):
        objects = pd.concat([_objects(actor_id=1), _objects(actor_id=2)], ignore_index=True)
        scored, _tracks = score_seeded_view(
            objects, "gt_seeded_motion_only", 2.0, 1.11, 20.0, [3]
        )
        summary = summarize_freshness(scored, [3])
        corpus = summary[(summary["scope"] == "corpus") & (summary["class_name"] == "all")]

        self.assertEqual(len(corpus), 1)
        self.assertEqual(int(corpus.iloc[0]["near_breach_3tick_frames"]), 3)
        self.assertEqual(int(corpus.iloc[0]["all_in_scope_frames"]), 5)


class DwellAndCensoringTests(unittest.TestCase):
    def test_dwell_is_split_by_time_gap(self):
        objects = _objects(timestamps=[0.0, 0.05, 0.10, 0.30, 0.35])
        dwell = dwell_segments(objects, 20.0, 2.0, [10.0])
        fast = dwell[dwell["regime"] == "fast_ge_10"]

        self.assertEqual(fast["object_frames"].tolist(), [3, 2])
        self.assertEqual(fast["dwell_s"].tolist(), [0.15, 0.10])

    def test_kaplan_meier_quantile_retains_censoring(self):
        self.assertEqual(_km_quantile([1.0, 2.0], [True, False], 0.50), 1.0)
        self.assertIsNone(_km_quantile([1.0, 2.0], [True, False], 0.90))

    def test_quantile_handles_stationary_infinite_budget(self):
        self.assertEqual(_quantile([1.0, 2.0, math.inf], 0.25), 1.5)
        self.assertTrue(math.isinf(_quantile([1.0, 2.0, math.inf], 0.90)))


if __name__ == "__main__":
    unittest.main()

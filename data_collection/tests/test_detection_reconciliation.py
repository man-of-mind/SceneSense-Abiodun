import unittest

import pandas as pd

from data_collection.reconcile_detection_coverage import (
    denominator_rows,
    mark_matches,
    pooled_summary,
    summarize_rows,
)


def _gt_row(actor_id, projected_x, bbox, distance=10.0):
    return {
        "frame_id": 1,
        "carla_timestamp": 0.1,
        "actor_id": actor_id,
        "class_name": "vehicle",
        "world_x": float(actor_id),
        "world_y": 0.0,
        "distance_m": distance,
        "in_camera_frustum": 1,
        "projected_x": projected_x,
        "projected_y": 100.0,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
    }


class DetectionReconciliationTests(unittest.TestCase):
    def test_offline_proxy_excludes_edge_center_and_tiny_box(self):
        gt = pd.DataFrame(
            [
                _gt_row(1, 100.0, (90.0, 90.0, 110.0, 110.0)),
                _gt_row(2, -1.0, (0.0, 90.0, 10.0, 110.0)),
                _gt_row(3, 100.0, (99.0, 99.0, 103.0, 103.0)),
            ]
        )

        current = denominator_rows(gt, "current_in_frustum_le25", 854, 480)
        proxy = denominator_rows(gt, "offline_visibility_proxy_le25", 854, 480)

        self.assertEqual(current["actor_id"].tolist(), [1, 2, 3])
        self.assertEqual(proxy["actor_id"].tolist(), [1])

    def test_matcher_retains_exact_live_metric_semantics(self):
        gt = pd.DataFrame([_gt_row(1, 100.0, (90.0, 90.0, 110.0, 110.0))])
        predictions = pd.DataFrame(
            [
                {
                    "frame_id": 1,
                    "class_name": "vehicle",
                    "world_x": 1.5,
                    "world_y": 0.0,
                    "distance_m": 10.0,
                    "score": 0.9,
                }
            ]
        )

        marked = mark_matches(gt, predictions)

        self.assertTrue(bool(marked.iloc[0]["matched"]))

    def test_pooled_summary_uses_counts_not_mean_of_run_percentages(self):
        metadata = {
            "corpus": "example",
            "episode_id": "run_a",
            "scenario_family": "family",
            "scenario_variant": "variant",
            "split": "test",
        }
        marked = pd.DataFrame(
            [
                {**_gt_row(1, 100.0, (90.0, 90.0, 110.0, 110.0)), "matched": True},
                {**_gt_row(2, 100.0, (90.0, 90.0, 110.0, 110.0)), "matched": False},
            ]
        )
        per_run = pd.DataFrame(summarize_rows(marked, metadata, "current_in_frustum_le25"))

        pooled = pooled_summary(per_run)
        vehicle = pooled[pooled["class_name"] == "vehicle"].iloc[0]

        self.assertEqual(int(vehicle["eligible_gt_rows"]), 2)
        self.assertEqual(float(vehicle["direct_object_row_coverage_pct"]), 50.0)


if __name__ == "__main__":
    unittest.main()

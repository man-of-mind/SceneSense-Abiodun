import unittest
import json
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data_collection.analyze_evaluation_contract import (
    _load_run,
    _match_array_count,
    choose_validation_thresholds,
)
from data_collection.verify_policy_corpus import (
    _trajectory_bootstrap_recall,
    verify,
)


class EvaluationContractAnalysisTests(unittest.TestCase):
    def test_report_only_bootstrap_uses_class_specific_ranges(self):
        per_run = pd.DataFrame(
            {
                "episode_id": ["p1", "v1"],
                "split": ["test", "test"],
                "contract": ["validation_f1", "validation_f1"],
                "class_name": ["pedestrian", "vehicle"],
                "eligible_gt_rows_le12m": [10, 0],
                "matched_gt_rows_le12m": [7, 0],
                "eligible_gt_rows_le25m": [20, 10],
                "matched_gt_rows_le25m": [12, 8],
            }
        )
        summary = _trajectory_bootstrap_recall(
            per_run,
            range_by_class={"pedestrian": 12.0, "vehicle": 25.0},
            samples=100,
            seed=7,
        ).set_index("class_name")
        self.assertAlmostEqual(summary.loc["pedestrian", "recall"], 0.7)
        self.assertAlmostEqual(summary.loc["vehicle", "recall"], 0.8)
        self.assertEqual(summary.loc["pedestrian", "range_upper_m"], 12.0)
        self.assertEqual(summary.loc["vehicle", "range_upper_m"], 25.0)

    def test_verifier_rejects_incomplete_full_batch_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            (batch_dir / "batch_manifest.json").write_text(
                json.dumps({"mode": "full", "status": "running"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not complete"):
                verify(batch_dir, skip_surrogate=True)

    def test_matching_is_one_to_one_inside_center_gate(self):
        gt_xy = np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=float)
        prediction_xy = np.asarray([[0.25, 0.0]], dtype=float)
        self.assertEqual(_match_array_count(gt_xy, prediction_xy), 1)

    def test_matching_rejects_centers_outside_five_metres(self):
        gt_xy = np.asarray([[0.0, 0.0]], dtype=float)
        prediction_xy = np.asarray([[5.01, 0.0]], dtype=float)
        self.assertEqual(_match_array_count(gt_xy, prediction_xy), 0)

    def test_ground_truth_matching_uses_actor_origin(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            streams = run_dir / "streams"
            streams.mkdir()
            pd.DataFrame(
                [
                    {
                        "frame_id": 1,
                        "class_name": "vehicle",
                        "world_x": 99.0,
                        "world_y": 99.0,
                        "origin_x": 1.0,
                        "origin_y": 2.0,
                        "distance_m": 10.0,
                        "in_camera_frustum": 1,
                    }
                ]
            ).to_csv(streams / "sample_object_ground_truth.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "frame_id": 1,
                        "class_name": "vehicle",
                        "world_x": 1.0,
                        "world_y": 2.0,
                        "distance_m": 10.0,
                        "score": 0.5,
                    }
                ]
            ).to_csv(streams / "sample_object_predictions.csv", index=False)
            run = _load_run(
                {
                    "episode_id": "episode",
                    "scenario_family": "family",
                    "split": "validation",
                    "run_dir": str(run_dir),
                }
            )
            self.assertEqual(float(run.gt.iloc[0]["world_x"]), 1.0)
            self.assertEqual(float(run.gt.iloc[0]["world_y"]), 2.0)

    def test_threshold_selection_uses_validation_f1_and_stricter_tie(self):
        rows = []
        for class_name in ("pedestrian", "vehicle"):
            rows.extend(
                [
                    {
                        "split": "validation",
                        "class_name": class_name,
                        "score_threshold": 0.10,
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                    },
                    {
                        "split": "validation",
                        "class_name": class_name,
                        "score_threshold": 0.20,
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                    },
                    {
                        "split": "test",
                        "class_name": class_name,
                        "score_threshold": 0.90,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                ]
            )
        selected = choose_validation_thresholds(pd.DataFrame(rows))
        self.assertEqual(set(selected["score_threshold"]), {0.20})


if __name__ == "__main__":
    unittest.main()

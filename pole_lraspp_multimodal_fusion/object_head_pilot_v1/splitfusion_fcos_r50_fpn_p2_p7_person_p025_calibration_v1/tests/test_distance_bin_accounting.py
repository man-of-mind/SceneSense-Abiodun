from __future__ import annotations

import math
import unittest

import torch

from ..distance_failure_audit import (
    BANDS,
    OVERFLOW_BAND,
    PREDICTION_BANDS,
    AuditError,
    band_of,
    score_stage,
)


def frame(sample_id: str, scores, world_xy, support) -> dict[str, torch.Tensor | str]:
    count = len(scores)
    return {
        "sample_id": sample_id,
        "experiment_id": "synthetic",
        "original_indices": torch.arange(count, dtype=torch.int32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "boxes": torch.zeros((count, 4), dtype=torch.float32),
        "world_xy": torch.tensor(world_xy, dtype=torch.float64),
        "component_ids": torch.full((count,), -1, dtype=torch.int32),
        "semantic_support": torch.tensor(support, dtype=torch.float32),
        "ignore_flags": torch.zeros((count,), dtype=torch.bool),
        "gt_world_xy": torch.zeros((0, 2), dtype=torch.float64),
        "semantic_component_count": 0,
    }


class DistanceBinAccountingCheck(unittest.TestCase):
    def test_band_edges_are_right_open_except_the_closed_40m_upper_edge(self) -> None:
        expected = [
            (0.0, "00_10m"), (9.999, "00_10m"), (10.0, "10_20m"), (19.999, "10_20m"),
            (20.0, "20_30m"), (29.999, "20_30m"), (30.0, "30_35m"), (34.999, "30_35m"),
            (35.0, "35_40m"), (39.999, "35_40m"), (40.0, "35_40m"),
        ]
        for distance, band in expected:
            self.assertEqual(band_of(distance, allow_overflow=False), band, distance)
            self.assertEqual(band_of(distance, allow_overflow=True), band, distance)
        self.assertEqual(band_of(40.001, allow_overflow=True), OVERFLOW_BAND)
        with self.assertRaises(AuditError):
            band_of(40.001, allow_overflow=False)
        with self.assertRaises(AuditError):
            band_of(-0.001, allow_overflow=True)
        self.assertEqual(PREDICTION_BANDS, tuple(name for name, _l, _u in BANDS) + (OVERFLOW_BAND,))

    def test_per_band_counts_sum_to_the_totals_under_split_gt_and_prediction_assignment(self) -> None:
        # One synthetic frame. The camera sits at the origin, so radial distance is |x|.
        # GT at 32 m is detected by a candidate whose own predicted distance lands in the
        # 35-40 m band, which separates the recall and precision assignments by construction.
        camera = {"synthetic_000": (0.0, 0.0)}
        candidates = [
            (5.0, 0.0),    # matches the 5 m GT
            (36.5, 0.0),   # too far from any observable GT; consumed by the AVO-ignore tier
            (32.4, 0.0),   # matches the 32 m GT, predicted band 30_35m
            (37.2, 0.0),   # unmatched -> FP in 35_40m
            (44.0, 0.0),   # unmatched -> FP in the overflow band
        ]
        item = frame(
            "synthetic_000",
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [list(point) for point in candidates],
            [0.5] * 5,
        )
        observable = {
            "synthetic_000": [
                {"world_x": 5.0, "world_y": 0.0, "distance_m": 5.0, "band": "00_10m"},
                {"world_x": 32.0, "world_y": 0.0, "distance_m": 32.0, "band": "30_35m"},
                {"world_x": 15.0, "world_y": 0.0, "distance_m": 15.0, "band": "10_20m"},
            ]
        }
        # The 36.5 m candidate is within 3 m of a 35-40 m AVO-ignored actor, so it must be
        # consumed by the AVO-ignore tier and never counted as a false positive.
        avo_ignored = {"synthetic_000": [{"world_x": 36.0, "world_y": 0.0}]}
        structural = {"synthetic_000": [{"world_x": 60.0, "world_y": 0.0}]}
        positions = {"synthetic_000": {"stage": torch.arange(5, dtype=torch.long)}}

        view = score_stage(
            frames=[item],
            stage_key="stage",
            stage_positions_by_sample=positions,
            observable_gt=observable,
            avo_ignored_gt=avo_ignored,
            structural_gt=structural,
            camera_xy_by_sample=camera,
        )
        overall, bands = view["overall"], view["bands"]
        self.assertTrue(all(view["accounting_checks"].values()))
        self.assertEqual(overall["eligible_gt"], 3)
        self.assertEqual(overall["tp_gt_band"], 2)
        self.assertEqual(overall["tp_pred_band"], 2)
        self.assertEqual(overall["fn"], 1)
        self.assertEqual(overall["fp"], 2)
        self.assertEqual(overall["avo_ignored_predictions"], 1)
        self.assertEqual(overall["structural_ignored_predictions"], 0)

        for field in ("eligible_gt", "tp_gt_band", "tp_pred_band", "fp", "fn"):
            self.assertEqual(
                sum(bands[name][field] for name in PREDICTION_BANDS), overall[field], field
            )
        # Recall is banded by GT distance: the 32 m GT counts in 30_35m.
        self.assertEqual(bands["30_35m"]["tp_gt_band"], 1)
        self.assertEqual(bands["10_20m"]["fn"], 1)
        self.assertEqual(bands["10_20m"]["recall"], 0.0)
        self.assertEqual(bands["00_10m"]["recall"], 1.0)
        # Precision is banded by predicted radial distance: the same TP lands in 30_35m,
        # while the two unmatched candidates land in 35_40m and the overflow band.
        self.assertEqual(bands["30_35m"]["tp_pred_band"], 1)
        self.assertEqual(bands["35_40m"]["fp"], 1)
        self.assertEqual(bands[OVERFLOW_BAND]["fp"], 1)
        self.assertEqual(bands["35_40m"]["eligible_gt"], 0)
        self.assertEqual(bands["35_40m"]["precision"], 0.0)
        self.assertEqual(bands[OVERFLOW_BAND]["precision"], 0.0)
        self.assertEqual(overall["precision"], 2 / 4)
        self.assertEqual(overall["recall"], 2 / 3)

        # Ceilings see the 10-20 m GT as unreachable and the other two as reachable.
        self.assertEqual(overall["reachable_gt"], 2)
        self.assertEqual(overall["max_matching_gt"], 2)
        self.assertIsNone(bands["10_20m"]["xy_mae_m"])
        self.assertEqual(bands["10_20m"]["candidate_recall_ceiling"], 0.0)
        self.assertAlmostEqual(bands["30_35m"]["xy_mae_m"], 0.4, places=9)
        self.assertAlmostEqual(overall["xy_mae_m"], 0.2, places=9)
        self.assertAlmostEqual(
            overall["f1"],
            2.0 * (2 / 4) * (2 / 3) / ((2 / 4) + (2 / 3)),
            places=12,
        )
        self.assertTrue(math.isfinite(overall["f1"]))


if __name__ == "__main__":
    unittest.main()

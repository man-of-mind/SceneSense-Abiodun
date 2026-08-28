#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

try:
    from .continuation_policy_v3 import catastrophic_regression, decorate, rank_key
except ImportError:
    from continuation_policy_v3 import catastrophic_regression, decorate, rank_key


HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "configs/expanded_continuation_v3.json").read_text())
TRAIN = json.loads((HERE / "configs/expanded_training_v2.json").read_text())
BASELINE = TRAIN["baseline"]


def record(epoch: int, **overrides):
    metrics = dict(BASELINE)
    metrics.update(overrides)
    return {
        "epoch": epoch,
        "label": f"epoch_{epoch}",
        "selection_order": epoch,
        "metrics": metrics,
        "vehicle_duplicate_fp": overrides.get("vehicle_duplicate_fp", BASELINE["vehicle_duplicate_fp"]),
        "all_metrics_finite": True,
        "checkpoint_state_integrity": True,
    }


class ContinuationPolicyTests(unittest.TestCase):
    def test_duplicate_count_is_not_catastrophic(self):
        candidate = record(20, vehicle_duplicate_fp=100000)
        self.assertTrue(all(catastrophic_regression(candidate, BASELINE, CONTRACT).values()))

    def test_person_f1_drop_beyond_point_03_is_catastrophic(self):
        candidate = record(20, person_f1=BASELINE["person_f1"] - 0.031)
        self.assertFalse(catastrophic_regression(candidate, BASELINE, CONTRACT)["person_f1_ge_baseline_minus_0_03"])

    def test_service_count_precedes_other_ranking_metrics(self):
        fewer = decorate(record(10, vehicle_f1=0.99, person_f1=0.99), BASELINE, CONTRACT)
        more = decorate(record(
            20,
            vehicle_precision=0.81,
            vehicle_recall=0.86,
            person_precision=0.81,
            person_recall=0.81,
            vehicle_f1=0.83,
            person_f1=0.81,
            vehicle_xy_mae_m=0.9,
            person_xy_mae_m=1.1,
            vehicle_iou=0.86,
            person_box_mask_iou=0.51,
            foreground_miou=0.68,
        ), BASELINE, CONTRACT)
        self.assertLess(rank_key(more, CONTRACT), rank_key(fewer, CONTRACT))

    def test_earlier_epoch_breaks_exact_tie(self):
        earlier = decorate(record(20), BASELINE, CONTRACT)
        later = decorate(record(30), BASELINE, CONTRACT)
        self.assertLess(rank_key(earlier, CONTRACT), rank_key(later, CONTRACT))


if __name__ == "__main__":
    unittest.main()

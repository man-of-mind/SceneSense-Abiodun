#!/usr/bin/env python3
"""Deterministic unit checks for the registered person-refinement policy."""

from __future__ import annotations

import unittest

from policy_v1 import deficit, dominates, material


class PolicyTests(unittest.TestCase):
    def test_deficit_is_zero_at_all_person_targets(self) -> None:
        self.assertEqual(deficit({
            "person_precision": 0.80, "person_recall": 0.80,
            "person_xy_mae_m": 1.20, "person_box_mask_iou": 0.50,
        }), 0.0)

    def test_continuous_deficit_penalizes_each_shortfall(self) -> None:
        value = deficit({
            "person_precision": 0.40, "person_recall": 0.60,
            "person_xy_mae_m": 1.80, "person_box_mask_iou": 0.25,
        })
        self.assertAlmostEqual(value, 1.75)

    def test_material_path_a_and_b_are_independent(self) -> None:
        base = {
            "person_f1": 0.50, "person_recall": 0.50, "person_precision": 0.54,
            "person_xy_mae_m": 1.35, "person_box_mask_iou": 0.45,
        }
        contract = {
            "A": {"person_f1_delta_min": 0.03, "person_recall_delta_min": 0.04,
                  "person_precision_delta_min": -0.01, "person_xy_improvement_min_m": 0.05},
            "B": {"person_f1_delta_min": 0.015, "person_xy_improvement_min_m": 0.10,
                  "person_iou_delta_min": 0.02},
        }
        candidate = {
            "person_f1": 0.535, "person_recall": 0.545, "person_precision": 0.531,
            "person_xy_mae_m": 1.29, "person_box_mask_iou": 0.45,
        }
        result = material(candidate, base, contract)
        self.assertTrue(result["A"]["pass"])
        self.assertFalse(result["B"]["pass"])
        self.assertTrue(result["pass"])

    def test_pareto_dominance_uses_all_registered_person_axes(self) -> None:
        strong = {"metrics": {
            "person_f1": 0.60, "person_recall": 0.60, "person_precision": 0.60,
            "person_xy_mae_m": 1.10, "person_box_mask_iou": 0.55,
        }}
        weak = {"metrics": {
            "person_f1": 0.59, "person_recall": 0.59, "person_precision": 0.59,
            "person_xy_mae_m": 1.11, "person_box_mask_iou": 0.54,
        }}
        self.assertTrue(dominates(strong, weak))
        self.assertFalse(dominates(weak, strong))


if __name__ == "__main__":
    unittest.main()

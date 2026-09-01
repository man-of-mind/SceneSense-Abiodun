from __future__ import annotations

import unittest

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.runtime import (
    apply_relational_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    calibrate_vehicle_scores,
)

from ..contract import CANONICAL_PERSON_THRESHOLD, calibration_at_raw_threshold


class RelationalP070SyntheticTests(unittest.TestCase):
    def test_calibration_maps_selected_boundary_to_fp32_point_20(self) -> None:
        mapped = calibration_at_raw_threshold()
        self.assertEqual(mapped.dtype, torch.float32)
        self.assertTrue(torch.equal(
            mapped, torch.tensor([CANONICAL_PERSON_THRESHOLD], dtype=torch.float32),
        ))

    def test_vehicle_behavior_and_non_score_fields_are_unchanged(self) -> None:
        detections = {
            "scores": torch.tensor([0.91, 0.77, 0.63, 0.42], dtype=torch.float32),
            "labels_internal": torch.tensor([0, 1, 2, 1], dtype=torch.long),
            "boxes": torch.tensor([
                [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0],
            ], dtype=torch.float32),
            "candidate_identity": torch.tensor([
                [0, 0, 10, 0], [0, 1, 20, 1], [0, 2, 30, 2], [0, 3, 40, 1],
            ], dtype=torch.long),
            "world_xyz": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        }
        person_indices = torch.tensor([1, 3], dtype=torch.long)
        person_scores = torch.tensor([0.25, 0.19], dtype=torch.float32)
        result, keep = apply_relational_service_policy(
            detections, person_indices, person_scores,
        )
        expected_keep = torch.tensor([0, 1, 2], dtype=torch.long)
        self.assertTrue(torch.equal(keep, expected_keep))
        for name, value in detections.items():
            if name != "scores":
                self.assertTrue(torch.equal(result[name], value.index_select(0, expected_keep)))
        vehicle_source = torch.tensor([0, 2], dtype=torch.long)
        vehicle_positions = torch.tensor([0, 2], dtype=torch.long)
        self.assertTrue(torch.equal(
            result["scores"].index_select(0, vehicle_positions),
            calibrate_vehicle_scores(detections["scores"].index_select(0, vehicle_source)),
        ))
        self.assertEqual(float(result["scores"][1]), float(person_scores[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

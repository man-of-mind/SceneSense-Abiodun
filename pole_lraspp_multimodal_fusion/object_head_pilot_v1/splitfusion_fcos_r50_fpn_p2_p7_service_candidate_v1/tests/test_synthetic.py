from __future__ import annotations

import copy
import unittest

import torch

from ..provenance import (
    FROZEN_CHECKPOINT_SHA256,
    PERSON_RULE,
    VEHICLE_BASE_THRESHOLD,
    _validate_person_result,
)
from ..runtime import apply_combined_service_policy, calibrate_vehicle_scores, combined_records


class ServiceCandidateSyntheticChecks(unittest.TestCase):
    def test_1_locked_vehicle_threshold_maps_to_canonical_fp32_score(self) -> None:
        threshold = torch.tensor([VEHICLE_BASE_THRESHOLD], dtype=torch.float32)
        calibrated = calibrate_vehicle_scores(threshold)
        self.assertEqual(calibrated.dtype, torch.float32)
        self.assertTrue(torch.equal(calibrated, torch.tensor([0.20], dtype=torch.float32)))

    def test_2_combined_policy_preserves_order_and_non_authorized_fields(self) -> None:
        semantic_logits = torch.zeros((1, 3, 432, 768), dtype=torch.float32)
        semantic_logits[0, 2, 10:30, 10:30] = 2.0
        semantic_logits[0, 2, 60:80, 60:80] = 2.0
        outputs = {"semantic_logits": semantic_logits}
        detections = {
            "scores": torch.tensor([VEHICLE_BASE_THRESHOLD, 0.90, 0.70, 0.80, 0.65]),
            "labels_internal": torch.tensor([0, 1, 0, 1, 1]),
            "boxes": torch.tensor([
                [100.0, 100.0, 120.0, 120.0], [10.0, 10.0, 30.0, 30.0],
                [130.0, 100.0, 150.0, 120.0], [11.0, 11.0, 29.0, 29.0],
                [60.0, 60.0, 80.0, 80.0],
            ]),
            "world_xyz": torch.tensor([
                [20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [30.0, 0.0, 0.0],
                [0.5, 0.0, 0.0], [10.0, 0.0, 0.0],
            ], dtype=torch.float64),
            "candidate_identity": torch.tensor([
                [0, 0, 10, 0], [0, 1, 11, 1], [0, 2, 12, 0],
                [0, 3, 13, 1], [0, 4, 14, 1],
            ]),
            "opaque_geometry": torch.arange(15, dtype=torch.float64).reshape(5, 3),
        }
        combined, original_indices = apply_combined_service_policy(outputs, detections)
        self.assertEqual(original_indices.tolist(), [0, 1, 2, 4])
        for name in detections.keys() - {"scores"}:
            self.assertTrue(torch.equal(combined[name], detections[name].index_select(0, original_indices)), name)
        retained_classes = combined["labels_internal"]
        person = retained_classes == 1
        vehicle = ~person
        selected_scores = detections["scores"].index_select(0, original_indices)
        self.assertTrue(torch.equal(combined["scores"][person], selected_scores[person]))
        self.assertTrue(torch.equal(
            combined["scores"][vehicle], calibrate_vehicle_scores(selected_scores[vehicle]),
        ))

        class FakeInfer:
            @staticmethod
            def record(result: dict[str, torch.Tensor], _row: dict[str, str], index: int) -> dict[str, object]:
                return {"prediction_index": index,
                        "candidate_identity": tuple(result["candidate_identity"][index].tolist())}

        class FakeBase:
            infer = FakeInfer()

        records = combined_records(FakeBase(), {"sample_id": "synthetic"}, combined, original_indices)
        self.assertEqual([row["prediction_index"] for row in records], [0, 1, 2, 4])
        self.assertEqual([row["candidate_identity"] for row in records], [
            tuple(value.tolist()) for value in detections["candidate_identity"].index_select(0, original_indices)
        ])

    def test_3_wrong_person_result_hash_status_or_configuration_fails_closed(self) -> None:
        valid = {
            "schema": "splitfusion_fcos_person_instance_consolidation_result_v1",
            "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "status": "holdout_feasible",
            "grid_configuration_count": 36,
            "holdout_evaluations": 1,
            "validation_or_test_accessed": False,
            "selected_fit": {**PERSON_RULE, "precision": 0.81, "recall": 0.82},
            "holdout": {**PERSON_RULE, "precision": 0.83, "recall": 0.84},
        }
        _validate_person_result(valid, actual_sha256="locked", expected_sha256="locked")
        with self.assertRaises(RuntimeError):
            _validate_person_result(valid, actual_sha256="wrong", expected_sha256="locked")
        for name, mutate in (
            ("status", lambda value: value.update(status="train_infeasible")),
            ("configuration", lambda value: value["selected_fit"].update(grid_index=26)),
        ):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                invalid = copy.deepcopy(valid)
                mutate(invalid)
                _validate_person_result(invalid, actual_sha256="locked", expected_sha256="locked")


if __name__ == "__main__":
    unittest.main()

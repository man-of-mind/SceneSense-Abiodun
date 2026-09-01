from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.runtime import (
    apply_relational_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    calibrate_vehicle_scores,
)

from ..contract import (
    CANONICAL_PERSON_THRESHOLD,
    DEPLOYMENT_LOGIT_BIAS,
    FROZEN_CHECKPOINT_SHA256,
    SELECTOR_CHECKPOINT_SHA256,
    calibration_at_raw_threshold,
)
from ..evaluate_relational_p070 import validate_prediction_directory


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

    def test_evaluator_accepts_only_the_relational_p070_completion_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction = Path(temporary)
            detections = prediction / "detections.csv"
            segmentation = prediction / "segmentation_manifest.csv"
            detections.write_bytes(b"sample_id,score\nsynthetic,0.2\n")
            segmentation.write_bytes(b"sample_id,prediction_path\nsynthetic,segmentation/synthetic.png\n")
            detection_hash = hashlib.sha256(detections.read_bytes()).hexdigest()
            segmentation_hash = hashlib.sha256(segmentation.read_bytes()).hexdigest()
            valid = {
                "schema": "splitfusion_fcos_relational_p070_inference_v1",
                "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
                "selector_checkpoint_sha256": SELECTOR_CHECKPOINT_SHA256,
                "historical_selector_status_unchanged": "train_infeasible",
                "revised_objective": {"precision": 0.70, "recall": 0.70},
                "deployment_bias": DEPLOYMENT_LOGIT_BIAS,
                "deployment_threshold": 0.20,
                "validation_frames": 3345,
                "inference_pass_count": 1,
                "candidate_creation": False,
                "nms_rerun": False,
                "candidate_order": "original_post_nms",
                "prediction_index": "original_post_nms",
                "consolidation_is_feature_only": True,
                "vehicle_behavior": "bit_exact_service_candidate_v1",
                "geometry_changed": False,
                "segmentation_changed": False,
                "detections_sha256": detection_hash,
                "segmentation_manifest_sha256": segmentation_hash,
                "prediction_set_sha256": hashlib.sha256(
                    (detection_hash + segmentation_hash).encode(),
                ).hexdigest(),
            }
            sentinel = prediction / "INFERENCE_COMPLETE"
            manifest = prediction / "inference_manifest.json"

            def write(payload: dict[str, object], completion: str) -> None:
                sentinel.write_text(completion, encoding="utf-8")
                manifest.write_text(json.dumps(payload), encoding="utf-8")

            write(valid, "RELATIONAL_P070_INFERENCE_COMPLETE\n")
            resolved, accepted = validate_prediction_directory(prediction)
            self.assertEqual(resolved, prediction.resolve())
            self.assertEqual(accepted, valid)

            with self.subTest("old completion sentinel"), self.assertRaises(RuntimeError):
                write(valid, "SERVICE_CANDIDATE_INFERENCE_COMPLETE\n")
                validate_prediction_directory(prediction)
            for name, field, value in (
                ("old schema", "schema", "splitfusion_fcos_service_candidate_inference_v1"),
                ("altered calibration", "deployment_bias", DEPLOYMENT_LOGIT_BIAS + 0.01),
            ):
                with self.subTest(name), self.assertRaises(RuntimeError):
                    altered = copy.deepcopy(valid)
                    altered[field] = value
                    write(altered, "RELATIONAL_P070_INFERENCE_COMPLETE\n")
                    validate_prediction_directory(prediction)


if __name__ == "__main__":
    unittest.main(verbosity=2)

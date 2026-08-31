from __future__ import annotations

import math
import unittest
from unittest import mock

import torch
from torch import nn

from ..train_verifier import _deployment_calibration_result
from ..verifier import (
    FEATURE_DIM,
    HOLDOUT_EXPERIMENT_IDS,
    ROI_DESCRIPTOR_DIM,
    SCALAR_FEATURE_NAMES,
    PersonRoIDescriptor,
    PersonVerifier,
    apply_person_refinement,
    build_verifier_optimizer,
    calibration_bias_for_interval,
    exact_pr_report,
    fp16_round_trip_roi_descriptors,
    partition_experiment_ids,
    refined_person_logits,
    refined_person_scores,
)


class PersonRoIVerifierSyntheticChecks(unittest.TestCase):
    def test_1_vectorized_roi_descriptor_shape_and_order(self) -> None:
        channel = torch.arange(256, dtype=torch.float32).view(1, 256, 1, 1)
        p2 = channel.expand(1, 256, 112, 192).contiguous()
        p3 = (channel + 1000.0).expand(1, 256, 56, 96).contiguous()
        features = {
            "p2": p2,
            "p3": p3,
            "p4": torch.zeros((1, 256, 1, 1)),
            "p5": torch.zeros((1, 256, 1, 1)),
            "p6": torch.zeros((1, 256, 1, 1)),
            "p7": torch.zeros((1, 256, 1, 1)),
        }
        semantic_logits = torch.zeros((1, 3, 432, 768))
        semantic_logits[:, 1].fill_(math.log(2.0))
        semantic_logits[:, 2].fill_(math.log(4.0))
        outputs = {"features": features, "semantic_logits": semantic_logits}
        levels = torch.tensor([0, 3])
        points = torch.zeros(2, dtype=torch.long)
        classes = torch.ones(2, dtype=torch.long)
        scores = torch.tensor([0.20, 0.70])
        boxes = torch.tensor([[0.0, 0.0, 32.0, 32.0], [100.0, 50.0, 500.0, 400.0]])
        detections = {
            "scores": scores,
            "level_indices": levels,
            "point_indices": points,
            "labels_internal": classes,
            "boxes": boxes,
            "candidate_identity": torch.stack((torch.zeros_like(levels), levels, points, classes), dim=1),
            "depth_bin_probabilities": torch.tensor([[0.75, 0.25], [0.50, 0.50]]),
        }
        extractor = PersonRoIDescriptor()
        with mock.patch.object(extractor.roi_align, "forward", wraps=extractor.roi_align.forward) as roi_forward:
            descriptors, scalars, indices = extractor(outputs, detections)
        roi_forward.assert_called_once()
        self.assertEqual(descriptors.shape, (2, ROI_DESCRIPTOR_DIM))
        self.assertEqual(scalars.shape, (2, len(SCALAR_FEATURE_NAMES)))
        self.assertEqual(indices.tolist(), [0, 1])
        torch.testing.assert_close(
            descriptors[0], torch.arange(256, dtype=torch.float32).repeat_interleave(4),
        )
        torch.testing.assert_close(
            descriptors[1], (torch.arange(256, dtype=torch.float32) + 1000.0).repeat_interleave(4),
        )
        torch.testing.assert_close(scalars[:, 0], scores)
        torch.testing.assert_close(scalars[:, 1], levels.float() / 5.0)
        torch.testing.assert_close(scalars[:, 2], torch.full((2,), 4.0 / 7.0))
        torch.testing.assert_close(scalars[:, 3], torch.full((2,), 4.0 / 7.0))
        torch.testing.assert_close(scalars[:, 4], torch.tensor([0.75, 0.50]))
        expected_entropy = torch.tensor([
            -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2.0),
            1.0,
        ])
        torch.testing.assert_close(scalars[:, 5], expected_entropy)
        torch.testing.assert_close(scalars[:, 6], torch.tensor([32.0 / 768.0, 400.0 / 768.0]))
        torch.testing.assert_close(scalars[:, 7], torch.tensor([32.0 / 448.0, 350.0 / 448.0]))
        torch.testing.assert_close(
            scalars[:, 8], torch.tensor([32.0 * 32.0 / (448.0 * 768.0), 400.0 * 350.0 / (448.0 * 768.0)]),
        )
        torch.testing.assert_close(scalars[:, 9], torch.tensor([0.0, math.log(400.0 / 350.0)]))

    def test_2_neutral_initialization_reproduces_person_scores(self) -> None:
        torch.manual_seed(2)
        head = PersonVerifier()
        features = torch.randn((4, FEATURE_DIM))
        delta = head(features)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        detections = {
            "labels_internal": torch.ones(4, dtype=torch.long),
            "scores": torch.tensor([0.02, 0.20, 0.50, 0.99]),
        }
        refined = apply_person_refinement(detections, delta, calibration_bias=0.0)
        torch.testing.assert_close(refined["scores"], detections["scores"], rtol=1e-6, atol=1e-7)

    def test_3_vehicle_records_and_scores_are_unchanged(self) -> None:
        classes = torch.tensor([0, 1, 0, 1])
        detections = {
            "labels_internal": classes,
            "scores": torch.tensor([0.12345678, 0.20, 0.87654321, 0.40]),
            "boxes": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "world_xyz": torch.arange(12, dtype=torch.float64).reshape(4, 3),
            "candidate_identity": torch.tensor([[0, 0, 1, 0], [0, 1, 2, 1], [0, 2, 3, 0], [0, 3, 4, 1]]),
        }
        refined = apply_person_refinement(detections, torch.tensor([0.4, -0.7]), calibration_bias=0.2)
        vehicle = classes == 0
        for name, value in detections.items():
            if value.shape[0] == classes.numel():
                self.assertTrue(torch.equal(refined[name][vehicle], value[vehicle]), name)
        self.assertTrue(torch.equal(refined["scores"][vehicle], detections["scores"][vehicle]))
        self.assertTrue(torch.equal(refined["boxes"], detections["boxes"]))
        self.assertTrue(torch.equal(refined["candidate_identity"], detections["candidate_identity"]))

    def test_4_exact_disjoint_eight_fit_two_holdout_episode_split(self) -> None:
        fit_candidates = [f"canonical_v3_{index:02d}_train_synthetic_{index}" for index in range(1, 9)]
        fit, holdout = partition_experiment_ids([*fit_candidates, *HOLDOUT_EXPERIMENT_IDS])
        self.assertEqual(len(fit), 8)
        self.assertEqual(len(holdout), 2)
        self.assertEqual(set(holdout), set(HOLDOUT_EXPERIMENT_IDS))
        self.assertTrue(set(fit).isdisjoint(holdout))

    def test_5_optimizer_contains_only_verifier_parameters(self) -> None:
        frozen_base = nn.Linear(4, 4)
        for parameter in frozen_base.parameters():
            parameter.requires_grad_(False)
        head = PersonVerifier()
        optimizer = build_verifier_optimizer(head)
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in head.parameters()})
        self.assertTrue(optimizer_ids.isdisjoint({id(parameter) for parameter in frozen_base.parameters()}))

    def test_6_training_and_inference_share_fp16_roi_representation(self) -> None:
        raw_roi = torch.linspace(-1.0, 1.0, 2 * ROI_DESCRIPTOR_DIM).reshape(2, ROI_DESCRIPTOR_DIM)
        cached_roi = raw_roi.to(torch.float16)
        training_roi = fp16_round_trip_roi_descriptors(cached_roi)
        inference_roi = fp16_round_trip_roi_descriptors(raw_roi)
        self.assertEqual(training_roi.dtype, torch.float32)
        self.assertTrue(torch.equal(training_roi, inference_roi))
        self.assertFalse(torch.equal(raw_roi, inference_roi))
        scalars = torch.tensor([[0.1234567] * len(SCALAR_FEATURE_NAMES)] * 2, dtype=torch.float32)
        base_scores = torch.tensor([0.2345678, 0.3456789], dtype=torch.float32)
        self.assertFalse(torch.equal(scalars, scalars.to(torch.float16).float()))
        self.assertFalse(torch.equal(base_scores, base_scores.to(torch.float16).float()))
        self.assertTrue(torch.equal(torch.cat((training_roi, scalars), dim=1)[:, -10:], scalars))
        self.assertTrue(torch.equal(base_scores, base_scores.float()))

    def test_7_fp32_calibration_recheck_fails_closed(self) -> None:
        boundary_logit = -33.41585159301758
        base_scores = torch.full((6,), 0.5, dtype=torch.float32)
        delta = torch.tensor([boundary_logit] * 4 + [-40.0] * 2, dtype=torch.float32)
        labels = torch.tensor([1, 1, 1, 1, 0, 0])
        before = exact_pr_report(refined_person_scores(base_scores, delta), labels, eligible_positive_count=4)
        interval = before["selected_interval"]
        self.assertIsNotNone(interval)
        attempted_bias = calibration_bias_for_interval(interval)
        old_fp64_scores = torch.sigmoid(refined_person_logits(base_scores, delta).double() + attempted_bias)
        old_metrics = exact_pr_report(old_fp64_scores, labels, eligible_positive_count=4)["at_0_20"]
        self.assertGreaterEqual(old_metrics["precision"], 0.80)
        self.assertGreaterEqual(old_metrics["recall"], 0.80)

        result = _deployment_calibration_result(base_scores, delta, labels, 4, interval)
        self.assertEqual(result["after_calibration"]["at_0_20"]["recall"], 0.0)
        self.assertEqual(result["calibration_bias"], 0.0)
        self.assertEqual(result["status"], "train_infeasible")
        self.assertFalse(result["validation_allowed"])


if __name__ == "__main__":
    unittest.main()

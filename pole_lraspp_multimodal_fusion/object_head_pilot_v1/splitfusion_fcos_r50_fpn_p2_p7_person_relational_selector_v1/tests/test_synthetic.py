from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.provenance import (
    VEHICLE_BASE_THRESHOLD,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    calibrate_vehicle_scores,
)

from ..cache_join import join_shard_payloads
from .. import runtime as selector_runtime
from ..provenance import (
    CONSOLIDATION_MANIFEST_SHA256,
    FIT_EXPERIMENT_IDS,
    FROZEN_CHECKPOINT_SHA256,
    HOLDOUT_EXPERIMENT_IDS,
    ROI_MANIFEST_SHA256,
)
from ..runtime import apply_relational_service_policy
from ..selector import (
    ARCHITECTURE,
    INPUT_DIM,
    PersonRelationalSelector,
    build_selector_optimizer,
    refined_person_scores,
)
from ..train_selector import ScoredHoldoutFrame, _rematched_metrics_at_threshold


def _synthetic_cache_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, int]]:
    scores = torch.tensor([0.30, 0.60, 0.80], dtype=torch.float32)
    descriptors = torch.linspace(-1.0, 1.0, 3 * 1024, dtype=torch.float32).reshape(3, 1024).half()
    scalars = torch.zeros((3, 10), dtype=torch.float32)
    scalars[:, 0] = scores
    roi = {
        "roi_descriptors": descriptors,
        "scalar_features": scalars,
        "base_scores": scores,
        "labels": torch.tensor([1, 0, 1]),
        "candidate_identities": torch.tensor([
            [0, 0, 10, 1], [0, 1, 11, 1], [0, 2, 12, 1],
        ]),
        "sample_ids": ["fit_frame", "fit_frame", "holdout_frame"],
        "experiment_ids": ["fit_episode", "fit_episode", "holdout_episode"],
        "partitions": torch.tensor([0, 0, 1], dtype=torch.int8),
    }
    consolidation = {"frames": [
        {
            "sample_id": "fit_frame",
            "experiment_id": "fit_episode",
            "original_indices": torch.tensor([1, 3], dtype=torch.int32),
            "scores": scores[:2].clone(),
            "boxes": torch.tensor([[0.0, 0.0, 76.8, 43.2], [100.0, 50.0, 150.0, 120.0]]),
            "world_xy": torch.tensor([[10.0, 2.0], [20.0, 4.0]], dtype=torch.float64),
            "component_ids": torch.tensor([1, 2], dtype=torch.int32),
            "semantic_support": torch.tensor([0.20, 0.30]),
            "ignore_flags": torch.tensor([False, False]),
            "gt_world_xy": torch.tensor([[10.0, 2.0]], dtype=torch.float64),
        },
        {
            "sample_id": "holdout_frame",
            "experiment_id": "holdout_episode",
            "original_indices": torch.tensor([0], dtype=torch.int32),
            "scores": scores[2:].clone(),
            "boxes": torch.tensor([[300.0, 100.0, 340.0, 180.0]]),
            "world_xy": torch.tensor([[5.0, -2.0]], dtype=torch.float64),
            "component_ids": torch.tensor([-1], dtype=torch.int32),
            "semantic_support": torch.tensor([0.0]),
            "ignore_flags": torch.tensor([False]),
            "gt_world_xy": torch.tensor([[5.0, -2.0]], dtype=torch.float64),
        },
    ]}
    return roi, consolidation, {"fit_episode": 0, "holdout_episode": 1}


class PersonRelationalSelectorSyntheticChecks(unittest.TestCase):
    def test_1_cache_join_is_exact_and_mismatches_fail_closed(self) -> None:
        roi, consolidation, partitions = _synthetic_cache_payloads()
        frames = join_shard_payloads(roi, consolidation, partitions, shard_name="synthetic")
        self.assertEqual([frame.sample_id for frame in frames], ["fit_frame", "holdout_frame"])
        self.assertEqual([frame.candidate_count for frame in frames], [2, 1])
        self.assertEqual([frame.features.shape for frame in frames], [(2, INPUT_DIM), (1, INPUT_DIM)])
        self.assertTrue(torch.equal(torch.cat([frame.base_scores for frame in frames]), roi["base_scores"]))
        self.assertTrue(torch.equal(frames[0].features[:, :1024], roi["roi_descriptors"][:2].float()))

        mutations = []
        wrong_sample = copy.deepcopy(roi)
        wrong_sample["sample_ids"][0] = "wrong"
        mutations.append((wrong_sample, consolidation))
        wrong_order = copy.deepcopy(consolidation)
        wrong_order["frames"][0]["scores"] = wrong_order["frames"][0]["scores"].flip(0)
        mutations.append((roi, wrong_order))
        wrong_partition = copy.deepcopy(roi)
        wrong_partition["partitions"][0] = 1
        mutations.append((wrong_partition, consolidation))
        for bad_roi, bad_consolidation in mutations:
            with self.subTest(), self.assertRaises(RuntimeError):
                join_shard_payloads(bad_roi, bad_consolidation, partitions, shard_name="synthetic")

    def test_2_selector_is_permutation_equivariant(self) -> None:
        torch.manual_seed(2)
        selector = PersonRelationalSelector().eval()
        nn.init.normal_(selector.output.weight, std=0.05)
        nn.init.normal_(selector.output.bias, std=0.05)
        features = torch.randn((1, 6, INPUT_DIM))
        padding = torch.zeros((1, 6), dtype=torch.bool)
        permutation = torch.tensor([3, 0, 5, 2, 1, 4])
        with torch.inference_mode():
            original = selector(features, padding)
            permuted = selector(features[:, permutation], padding[:, permutation])
        torch.testing.assert_close(permuted, original[:, permutation], rtol=1e-5, atol=1e-6)

    def test_3_padding_mask_isolates_real_candidates(self) -> None:
        torch.manual_seed(3)
        selector = PersonRelationalSelector().eval()
        nn.init.normal_(selector.output.weight, std=0.05)
        prefix = torch.randn((1, 3, INPUT_DIM))
        first = torch.cat((prefix, torch.zeros((1, 2, INPUT_DIM))), dim=1)
        second = torch.cat((prefix, torch.randn((1, 2, INPUT_DIM)) * 1000.0), dim=1)
        padding = torch.tensor([[False, False, False, True, True]])
        with torch.inference_mode():
            first_output = selector(first, padding)
            second_output = selector(second, padding)
        torch.testing.assert_close(first_output[:, :3], second_output[:, :3], rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(first_output[:, 3:], torch.zeros((1, 2))))
        self.assertTrue(torch.equal(second_output[:, 3:], torch.zeros((1, 2))))

    def test_4_neutral_initialization_exactly_reproduces_base_scores(self) -> None:
        torch.manual_seed(4)
        selector = PersonRelationalSelector().eval()
        features = torch.randn((1, 4, INPUT_DIM))
        padding = torch.zeros((1, 4), dtype=torch.bool)
        base_scores = torch.tensor([[0.02, 0.20, 0.50, 0.99]], dtype=torch.float32)
        with torch.inference_mode():
            residual = selector(features, padding)
            scores = refined_person_scores(base_scores, residual)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(scores, base_scores))

    def test_5_optimizer_ownership_and_service_outputs_are_restricted(self) -> None:
        frozen_base = nn.Linear(4, 4)
        frozen_base.requires_grad_(False)
        selector = PersonRelationalSelector()
        optimizer = build_selector_optimizer(selector)
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in selector.parameters()})
        self.assertTrue(optimizer_ids.isdisjoint({id(parameter) for parameter in frozen_base.parameters()}))

        detections = {
            "scores": torch.tensor([VEHICLE_BASE_THRESHOLD, 0.40, 0.70, 0.30], dtype=torch.float32),
            "labels_internal": torch.tensor([0, 1, 0, 1]),
            "boxes": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "world_xyz": torch.arange(12, dtype=torch.float64).reshape(4, 3),
            "dimensions": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "yaw": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
            "candidate_identity": torch.tensor([
                [0, 0, 1, 0], [0, 1, 2, 1], [0, 2, 3, 0], [0, 3, 4, 1],
            ]),
        }
        result, keep = apply_relational_service_policy(
            detections, torch.tensor([1, 3]), torch.tensor([0.80, 0.10]),
        )
        self.assertEqual(keep.tolist(), [0, 1, 2])
        for name in detections.keys() - {"scores"}:
            self.assertTrue(torch.equal(result[name], detections[name].index_select(0, keep)), name)
        vehicle_positions = torch.tensor([0, 2])
        self.assertTrue(torch.equal(
            result["scores"].index_select(0, vehicle_positions),
            calibrate_vehicle_scores(detections["scores"].index_select(0, vehicle_positions)),
        ))
        self.assertEqual(float(result["scores"][1]), float(torch.tensor(0.80, dtype=torch.float32)))

    def test_6_holdout_threshold_uses_canonical_rematching_not_cached_labels(self) -> None:
        cached_labels = torch.tensor([1, 0])
        frame = ScoredHoldoutFrame(
            sample_id="synthetic",
            experiment_id="synthetic_holdout",
            original_indices=torch.tensor([4, 9]),
            boxes=torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 10.0, 40.0, 20.0]]),
            world_xy=torch.tensor([[0.5, 0.0], [2.0, 0.0]], dtype=torch.float64),
            ignore_flags=torch.tensor([False, False]),
            gt_world_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            base_scores=torch.tensor([0.90, 0.80]),
            residual_logits=torch.zeros(2),
            refined_scores=torch.tensor([0.10, 0.80]),
        )
        self.assertEqual(cached_labels[torch.tensor([False, True])].tolist(), [0])
        metrics = _rematched_metrics_at_threshold([frame], [frame.refined_scores], 0.50)
        self.assertEqual(
            {name: metrics["aggregate"][name] for name in ("tp", "fp", "fn", "ignored")},
            {"tp": 1, "fp": 0, "fn": 0, "ignored": 0},
        )

    def test_7_checkpoint_rejects_mismatched_manifest_hash(self) -> None:
        lower, upper = 0.40, 0.60
        midpoint_logit = 0.5 * (
            math.log(lower / (1.0 - lower)) + math.log(upper / (1.0 - upper))
        )
        selected_threshold = 1.0 / (1.0 + math.exp(-midpoint_logit))
        calibration_bias = math.log(0.20 / 0.80) - midpoint_logit
        interval = {
            "lower_score_exclusive": lower,
            "upper_score_inclusive": upper,
            "midpoint_logit": midpoint_logit,
            "selected_threshold": selected_threshold,
        }

        def metric(threshold: float, true_positive: int) -> dict[str, float | int]:
            return {
                "threshold": threshold, "tp": true_positive, "fp": 0, "fn": 0, "ignored": 1,
                "precision": 1.0, "recall": 1.0,
            }

        selected = {
            "aggregate": {**metric(selected_threshold, 16), "ignored": 2},
            "episodes": {name: metric(selected_threshold, 8) for name in HOLDOUT_EXPERIMENT_IDS},
        }
        deployment = {
            "aggregate": {**metric(0.20, 16), "ignored": 2},
            "episodes": {name: metric(0.20, 8) for name in HOLDOUT_EXPERIMENT_IDS},
        }
        checkpoint = {
            "schema": "splitfusion_fcos_person_relational_selector_v1",
            "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "roi_manifest_sha256": ROI_MANIFEST_SHA256,
            "consolidation_manifest_sha256": CONSOLIDATION_MANIFEST_SHA256,
            "architecture": ARCHITECTURE,
            "selector": PersonRelationalSelector().state_dict(),
            "training": {
                "epochs": 5, "selected_epoch": 5, "batch_frames": 16, "optimizer": "Adam",
                "learning_rate": 1e-3, "positive_to_negative_loss_sampling": "1:3",
                "all_candidates_retained_in_attention_context": True,
                "ignored_labels_excluded_from_loss": True, "sampling_plan_scans": 1,
                "fit_episodes": list(FIT_EXPERIMENT_IDS),
                "holdout_episodes": list(HOLDOUT_EXPERIMENT_IDS), "seed": 20260831,
                "epoch_losses": [0.5] * 5,
                "epoch_sampling": [{"positive": 2, "negative": 6}] * 5,
            },
            "holdout": {
                "threshold_source": "one fixed epoch and one joint two-episode holdout threshold",
                "before_calibration": {
                    "candidate_scores_computed_once": True,
                    "tie_processing": "all_equal_scores_added_before_affected_frames_are_rematched",
                    "score_boundaries": 2,
                    "joint_precision_recall_0_80_exists": True,
                    "selected_interval": interval,
                },
                "joint_feasible_interval": interval,
                "selected_threshold_metrics": selected,
                "attempted_calibration_bias": calibration_bias,
                "calibration_bias": calibration_bias,
                "deployment_at_0_20": deployment,
                "selected_deployment_counts_agree": True,
            },
            "status": "train_feasible",
            "validation_allowed": True,
            "validation_or_test_accessed": False,
        }
        with mock.patch.object(selector_runtime.torch, "load", return_value=checkpoint):
            loaded, loaded_bias = selector_runtime.load_selector_checkpoint(Path(__file__), torch.device("cpu"))
        self.assertIsInstance(loaded, PersonRelationalSelector)
        self.assertEqual(loaded_bias, calibration_bias)

        invalid = copy.deepcopy(checkpoint)
        invalid["roi_manifest_sha256"] = "wrong"
        with mock.patch.object(selector_runtime.torch, "load", return_value=invalid):
            with self.assertRaises(RuntimeError):
                selector_runtime.load_selector_checkpoint(Path(__file__), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

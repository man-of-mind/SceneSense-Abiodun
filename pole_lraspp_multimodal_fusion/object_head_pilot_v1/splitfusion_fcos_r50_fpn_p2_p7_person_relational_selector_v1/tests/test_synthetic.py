from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.provenance import (
    VEHICLE_BASE_THRESHOLD,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    calibrate_vehicle_scores,
)

from ..cache_join import join_shard_payloads
from ..runtime import apply_relational_service_policy
from ..selector import (
    INPUT_DIM,
    PersonRelationalSelector,
    build_selector_optimizer,
    refined_person_scores,
)


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


if __name__ == "__main__":
    unittest.main()

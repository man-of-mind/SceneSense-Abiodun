from __future__ import annotations

import unittest

import torch
from torch import nn

from ..labeling import label_candidates
from ..quality import FEATURE_DIM, QualityMLP, build_quality_optimizer, refine_scores


class CandidateQualitySyntheticChecks(unittest.TestCase):
    def test_1_neutral_initialization_reproduces_base_score(self) -> None:
        torch.manual_seed(1)
        head = QualityMLP(normalize=False)
        features = torch.randn(8, FEATURE_DIM)
        base_scores = torch.tensor([0.02, 0.05, 0.20, 0.31, 0.50, 0.73, 0.90, 0.99])
        refined = refine_scores(base_scores, head(features))
        torch.testing.assert_close(refined, base_scores, rtol=1e-6, atol=1e-7)

    def test_2_one_to_one_candidate_labels(self) -> None:
        candidate_world = torch.tensor([
            [0.1, 0.0],   # nearest correct vehicle
            [0.8, 0.0],   # duplicate vehicle
            [0.0, 0.0],   # person prediction on vehicle: background for its class
            [10.0, 0.0],  # vehicle prediction more than 3 m from vehicle GT
            [10.1, 0.0],  # geometrically correct person, but ignore-centred
        ])
        classes = torch.tensor([0, 0, 1, 0, 1])
        boxes = torch.tensor([
            [0.0, 0.0, 2.0, 2.0],
            [2.0, 0.0, 4.0, 2.0],
            [4.0, 0.0, 6.0, 2.0],
            [6.0, 0.0, 8.0, 2.0],
            [8.0, 0.0, 10.0, 2.0],
        ])
        gt_world = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        gt_classes = torch.tensor([0, 1])
        ignore = torch.zeros((4, 12), dtype=torch.bool)
        ignore[1, 9] = True
        labels, summary = label_candidates(
            candidate_world_xy=candidate_world, candidate_classes=classes, candidate_boxes=boxes,
            gt_world_xy=gt_world, gt_classes=gt_classes, ignore_mask=ignore,
        )
        self.assertEqual(labels.tolist(), [1, 0, 0, 0, -1])
        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["fn"], 1)
        self.assertEqual(summary["tp"] + summary["fn"], summary["eligible_gt"])
        self.assertTrue(summary["tp_plus_fn_reconciles"])

    def test_3_optimizer_contains_only_quality_head_parameters(self) -> None:
        frozen = nn.Linear(4, 256)
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        head = QualityMLP(normalize=False)
        optimizer = build_quality_optimizer(head)
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in head.parameters()})
        self.assertTrue(optimizer_ids.isdisjoint({id(parameter) for parameter in frozen.parameters()}))

    def test_4_optimizer_step_leaves_frozen_parameters_unchanged(self) -> None:
        torch.manual_seed(4)
        frozen = nn.Linear(4, 256)
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        before = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
        head = QualityMLP(normalize=False)
        optimizer = build_quality_optimizer(head)
        with torch.no_grad():
            frozen_features = frozen(torch.randn(6, 4))
        candidate_features = torch.cat((frozen_features, torch.randn(6, FEATURE_DIM - 256)), dim=1)
        loss = head(candidate_features).square().mean() + head(candidate_features).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for name, value in frozen.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from ..labeling import label_candidates
from ..quality import FEATURE_DIM, LEVEL_NAMES, QualityMLP, build_quality_optimizer, extract_candidate_features, refine_scores


class CandidateQualitySyntheticChecks(unittest.TestCase):
    def test_1_neutral_initialization_reproduces_base_score(self) -> None:
        torch.manual_seed(1)
        levels = torch.arange(6)
        classes = levels.remainder(2)
        base_scores = torch.tensor([0.02, 0.05, 0.20, 0.50, 0.73, 0.99])
        semantic_logits = torch.zeros((1, 3, 432, 768))
        semantic_logits[:, 1].fill_(math.log(2.0))
        semantic_logits[:, 2].fill_(math.log(4.0))
        outputs = {
            "features": {
                name: torch.full((1, 256, 1, 1), float(index + 1))
                for index, name in enumerate(LEVEL_NAMES)
            },
            "semantic_logits": semantic_logits,
        }
        detections = {
            "level_indices": levels,
            "point_indices": torch.zeros(6, dtype=torch.long),
            "labels_internal": classes,
            "scores": base_scores,
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]] * 6),
            "candidate_identity": torch.stack((torch.zeros_like(levels), levels, torch.zeros_like(levels), classes), dim=1),
            "depth_bin_probabilities": torch.full((6, 33), 1.0 / 33.0),
        }
        features = extract_candidate_features(outputs, detections)
        self.assertEqual(features.shape, (6, FEATURE_DIM))
        for level_index in range(6):
            torch.testing.assert_close(features[level_index, :256], torch.full((256,), float(level_index + 1)))
        torch.testing.assert_close(features[:, 256], base_scores)
        torch.testing.assert_close(features[:, 257], classes.float())
        torch.testing.assert_close(features[:, 258], levels.float() / 5.0)
        semantic_expected = torch.where(classes == 0, 2.0 / 7.0, 4.0 / 7.0)
        torch.testing.assert_close(features[:, 259], semantic_expected)
        torch.testing.assert_close(features[:, 260], semantic_expected)
        torch.testing.assert_close(features[:, 261], torch.full((6,), 1.0 / 33.0))
        torch.testing.assert_close(features[:, 262], torch.ones(6))
        head = QualityMLP(normalize=False)
        refined = refine_scores(base_scores, head(features))
        torch.testing.assert_close(refined, base_scores, rtol=1e-6, atol=1e-7)

    def test_2_one_to_one_candidate_labels(self) -> None:
        candidate_world = torch.tensor([
            [0.1, 0.0],   # nearest correct vehicle
            [0.8, 0.0],   # duplicate vehicle
            [0.0, 0.0],   # person prediction on vehicle: background for its class
            [10.0, 0.0],  # vehicle prediction more than 3 m from vehicle GT
            [10.1, 0.0],  # matched person centred in ignore: remains positive
            [30.0, 30.0],  # unmatched ignore-centred person: ignored
        ])
        classes = torch.tensor([0, 0, 1, 0, 1, 1])
        boxes = torch.tensor([
            [0.0, 0.0, 2.0, 2.0],
            [2.0, 0.0, 4.0, 2.0],
            [4.0, 0.0, 6.0, 2.0],
            [6.0, 0.0, 8.0, 2.0],
            [8.0, 0.0, 10.0, 2.0],
            [10.0, 0.0, 12.0, 2.0],
        ])
        gt_world = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        gt_classes = torch.tensor([0, 1])
        ignore = torch.zeros((4, 14), dtype=torch.bool)
        ignore[1, 1] = True
        ignore[1, 9] = True
        ignore[1, 11] = True
        labels, summary = label_candidates(
            candidate_world_xy=candidate_world, candidate_classes=classes, candidate_boxes=boxes,
            gt_world_xy=gt_world, gt_classes=gt_classes, ignore_mask=ignore,
        )
        self.assertEqual(labels.tolist(), [1, 0, 0, 0, 1, -1])
        self.assertEqual(summary["tp"], 2)
        self.assertEqual(summary["fn"], 0)
        self.assertEqual(summary["ignored"], 1)
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

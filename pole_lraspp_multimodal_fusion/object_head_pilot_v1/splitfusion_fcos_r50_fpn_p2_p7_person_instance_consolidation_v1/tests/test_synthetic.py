from __future__ import annotations

import unittest

import torch

from ..core import (
    HOLDOUT_EXPERIMENT_IDS,
    assign_components,
    connected_person_components,
    consolidate_person_candidates,
    evaluate_frames,
    partition_frames,
    rematch_person_frame,
)


class PersonInstanceConsolidationSyntheticChecks(unittest.TestCase):
    def test_1_component_assignment_uses_maximum_mask_iou(self) -> None:
        mask = torch.zeros((8, 8), dtype=torch.bool)
        mask[1:3, 1:3] = True
        mask[5:7, 5:7] = True
        components, count = connected_person_components(mask)
        boxes = torch.tensor([
            [1.0, 1.0, 3.0, 3.0],
            [4.0, 4.0, 7.0, 7.0],
            [0.0, 5.0, 2.0, 7.0],
        ])
        component_ids, support = assign_components(components, boxes)
        self.assertEqual(count, 2)
        self.assertEqual(component_ids.tolist(), [1, 2, -1])
        torch.testing.assert_close(support, torch.tensor([1.0, 4.0 / 9.0, 0.0]))

    def test_2_background_candidate_is_rejected_only_when_support_is_enabled(self) -> None:
        inputs = {
            "scores": torch.tensor([0.9, 0.8]),
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 6.0, 6.0]]),
            "world_xy": torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
            "component_ids": torch.tensor([-1, 1]),
            "semantic_support": torch.tensor([0.0, 0.5]),
            "original_indices": torch.tensor([0, 1]),
            "group_box_iou_threshold": None,
        }
        off = consolidate_person_candidates(**inputs, semantic_support_threshold=None)
        enabled = consolidate_person_candidates(**inputs, semantic_support_threshold=0.01)
        self.assertEqual(off.tolist(), [0, 1])
        self.assertEqual(enabled.tolist(), [1])

    def test_3_duplicate_group_keeps_deterministic_highest_score_winner(self) -> None:
        retained = consolidate_person_candidates(
            scores=torch.tensor([0.9, 0.9, 0.7]),
            boxes=torch.tensor([
                [0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 4.5, 4.5], [0.0, 0.0, 4.0, 4.0],
            ]),
            world_xy=torch.tensor([[0.0, 0.0], [0.5, 0.0], [10.0, 0.0]]),
            component_ids=torch.tensor([1, 1, 1]),
            semantic_support=torch.tensor([0.5, 0.5, 0.5]),
            original_indices=torch.tensor([0, 1, 2]),
            semantic_support_threshold=None,
            group_box_iou_threshold=0.10,
        )
        self.assertEqual(retained.tolist(), [0, 2])

    def test_4_post_filter_rematching_and_episode_isolation_are_exact(self) -> None:
        base_frame = {
            "sample_id": "synthetic",
            "original_indices": torch.tensor([0, 1], dtype=torch.int32),
            "scores": torch.tensor([0.9, 0.8]),
            "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 4.0, 2.0]]),
            "world_xy": torch.tensor([[0.1, 0.0], [0.5, 0.0]], dtype=torch.float64),
            "component_ids": torch.tensor([1, 2], dtype=torch.int32),
            "semantic_support": torch.tensor([0.0, 1.0]),
            "ignore_flags": torch.tensor([False, False]),
            "gt_world_xy": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            "semantic_component_count": 2,
        }
        original_labels, _summary = rematch_person_frame(base_frame, torch.tensor([0, 1]))
        self.assertEqual(original_labels.tolist(), [1, 0])
        configuration = {
            "grid_index": 6,
            "semantic_support_threshold": 0.01,
            "group_box_iou_threshold": None,
        }
        rematched = evaluate_frames([{**base_frame, "experiment_id": "canonical_v3_01_train_synthetic"}], configuration)
        self.assertEqual((rematched["tp"], rematched["fp"], rematched["fn"]), (1, 0, 0))

        fit_ids = [f"canonical_v3_{index:02d}_train_synthetic_{index}" for index in range(1, 9)]
        frames = [{**base_frame, "experiment_id": experiment_id} for experiment_id in (
            *fit_ids, *sorted(HOLDOUT_EXPERIMENT_IDS),
        )]
        fit, holdout, registered_fit, registered_holdout = partition_frames(frames)
        self.assertEqual((len(registered_fit), len(registered_holdout)), (8, 2))
        self.assertTrue(set(registered_fit).isdisjoint(registered_holdout))
        self.assertEqual((len(fit), len(holdout)), (8, 2))
        self.assertEqual(evaluate_frames(fit, configuration)["tp"], 8)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from ..policy import filter_consolidated_person_outputs


class PersonP025PolicySyntheticCheck(unittest.TestCase):
    def test_p025_is_exact_p020_subset_and_vehicles_are_bitwise_unchanged(self) -> None:
        detections = {
            "scores": torch.tensor([0.11, 0.20, 0.24, 0.25, 0.91], dtype=torch.float32),
            "labels_internal": torch.tensor([0, 1, 1, 1, 0], dtype=torch.int64),
            "boxes": torch.arange(20, dtype=torch.float32).reshape(5, 4),
            "world_xyz": torch.arange(15, dtype=torch.float64).reshape(5, 3),
            "opaque_identity": torch.tensor([[8, 0], [8, 1], [8, 2], [8, 3], [8, 4]]),
        }
        filtered, keep = filter_consolidated_person_outputs(detections)
        self.assertEqual(keep.tolist(), [0, 3, 4])
        for name, value in detections.items():
            self.assertTrue(torch.equal(filtered[name], value.index_select(0, keep)), name)
        original_vehicle = torch.tensor([0, 4])
        filtered_vehicle = torch.where(filtered["labels_internal"] != 1)[0]
        for name, value in detections.items():
            self.assertTrue(torch.equal(
                filtered[name].index_select(0, filtered_vehicle),
                value.index_select(0, original_vehicle),
            ), name)


if __name__ == "__main__":
    unittest.main()

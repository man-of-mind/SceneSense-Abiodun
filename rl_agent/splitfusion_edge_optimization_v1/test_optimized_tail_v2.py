"""Focused regressions for the synchronization-light v2 tail candidate."""

from __future__ import annotations

import unittest

import torch

from .optimized_tail import apply_p025_service_policy_optimized, tree_bitwise_equal
from .optimized_tail_v2 import (
    apply_p025_service_policy_v2,
    require_tree_finite_batched,
)


class OptimizedTailV2Tests(unittest.TestCase):
    def test_batched_finite_check_preserves_tensor_count_and_rejection(self) -> None:
        finite = {
            "left": torch.tensor([1.0, -2.0]),
            "nested": [torch.tensor([3], dtype=torch.int64), torch.empty(0)],
        }
        self.assertEqual(require_tree_finite_batched(finite, "finite"), 3)
        with self.assertRaisesRegex(Exception, "non-finite"):
            require_tree_finite_batched(
                {"bad": torch.tensor([float("nan")])}, "non-finite tree"
            )
        with self.assertRaisesRegex(Exception, "contains non-finite"):
            require_tree_finite_batched(
                {"finite": torch.tensor([1.0])},
                "invalid physical output",
                extra_checks=(torch.tensor(False),),
            )

    def test_p025_v2_is_bit_identical_to_qualified_v1(self) -> None:
        generator = torch.Generator().manual_seed(20260910)
        for _iteration in range(40):
            count = 20 + int(torch.randint(0, 61, (), generator=generator))
            labels = torch.randint(0, 2, (count,), generator=generator)
            scores = torch.rand((count,), generator=generator, dtype=torch.float32)
            left_top = torch.rand((count, 2), generator=generator) * torch.tensor(
                [700.0, 390.0]
            )
            extent = 2.0 + torch.rand((count, 2), generator=generator) * 60.0
            boxes = torch.cat((left_top, left_top + extent), dim=1).float()
            detections = {
                "boxes": boxes,
                "scores": scores,
                "labels_internal": labels,
                "world_xyz": torch.randn(
                    (count, 3), generator=generator, dtype=torch.float64
                ),
                "local_xyz": torch.randn(
                    (count, 3), generator=generator, dtype=torch.float32
                ),
            }
            logits = torch.randn(
                (1, 3, 432, 768), generator=generator, dtype=torch.float32
            )
            expected, expected_indices = apply_p025_service_policy_optimized(
                {"semantic_logits": logits}, detections
            )
            observed, observed_indices = apply_p025_service_policy_v2(
                {"semantic_logits": logits}, detections
            )
            self.assertTrue(tree_bitwise_equal(expected, observed))
            self.assertTrue(torch.equal(expected_indices, observed_indices))


if __name__ == "__main__":
    unittest.main()

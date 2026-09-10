"""Focused CPU regressions for the output-preserving classical policy path."""

from __future__ import annotations

import unittest

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    assign_components,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)

from .optimized_tail import (
    _assign_components_numpy,
    apply_p025_service_policy_optimized,
    tree_bitwise_equal,
)


class OptimizedTailTests(unittest.TestCase):
    def test_numpy_component_assignment_is_exact(self) -> None:
        generator = torch.Generator().manual_seed(20260909)
        for _iteration in range(40):
            labels = torch.randint(0, 9, (48, 80), generator=generator)
            left_top = torch.rand((50, 2), generator=generator) * torch.tensor(
                [70.0, 40.0]
            )
            extent = 1.0 + torch.rand((50, 2), generator=generator) * 15.0
            boxes = torch.cat((left_top, left_top + extent), dim=1)
            expected = assign_components(labels, boxes)
            observed = _assign_components_numpy(labels, boxes)
            self.assertTrue(torch.equal(expected[0], observed[0]))
            self.assertTrue(torch.equal(expected[1], observed[1]))

    def test_optimized_p025_policy_is_bit_identical(self) -> None:
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
            expected, expected_indices = apply_p025_service_policy(
                {"semantic_logits": logits}, detections
            )
            observed, observed_indices = apply_p025_service_policy_optimized(
                {"semantic_logits": logits}, detections
            )
            self.assertTrue(tree_bitwise_equal(expected, observed))
            self.assertTrue(torch.equal(expected_indices, observed_indices))


if __name__ == "__main__":
    unittest.main()

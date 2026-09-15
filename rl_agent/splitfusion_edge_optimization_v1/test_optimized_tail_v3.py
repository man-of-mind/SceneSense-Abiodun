"""Focused regressions for the overlapped v3 edge candidate."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    connected_person_components,
    person_mask_from_logits,
)

from .optimized_tail import (
    CONTENT_H,
    CONTENT_W,
    apply_p025_service_policy_optimized,
    tree_bitwise_equal,
)
from .optimized_tail_v3 import (
    _DeferredFiniteValidator,
    _connected_person_components_array,
    apply_p025_service_policy_v3,
)
from . import codec_v3
from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES, StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    InnerIdentity,
    InspectedInnerPayload,
)


class OptimizedTailV3Tests(unittest.TestCase):
    @staticmethod
    def _noae_payload() -> InspectedInnerPayload:
        return InspectedInnerPayload(
            identity=InnerIdentity(
                family="noAE",
                family_id=0,
                quantizer="UINT8",
                bit_width=8,
                q_e4=0,
                keep_count=21504,
                routing_tag=0,
                transported_channels=256,
                latent_width=None,
                wire_magic_ascii="test",
                wire_codec_id=0,
                wire_version=0,
            ),
            compressed_bytes=1,
            uncompressed_bytes=1,
            kind="noae_uint8",
            sparse_bytes=b"x",
            parsed=None,
        )

    def test_single_finite_codec_remains_fail_closed(self) -> None:
        codec = codec_v3.SingleFiniteCheckProductionSplitCodec()
        finite = torch.zeros((256, 112, 192), dtype=torch.float32)
        with mock.patch.object(
            codec_v3.uint8_codec, "decode", return_value=(finite, 0.0)
        ):
            decoded = codec.decode(
                self._noae_payload(),
                decoder=None,
                tail_device=torch.device("cpu"),
                timing=StageRecorder(EDGE_STAGES),
            )
        self.assertTrue(decoded.finite)
        self.assertIs(decoded.c2, finite)

        corrupted = finite.clone()
        corrupted[0, 0, 0] = float("nan")
        with mock.patch.object(
            codec_v3.uint8_codec, "decode", return_value=(corrupted, 0.0)
        ):
            with self.assertRaisesRegex(Exception, "non-finite"):
                codec.decode(
                    self._noae_payload(),
                    decoder=None,
                    tail_device=torch.device("cpu"),
                    timing=StageRecorder(EDGE_STAGES),
                )

    def test_direct_component_array_preserves_component_partition(self) -> None:
        mask = torch.zeros((CONTENT_H, CONTENT_W), dtype=torch.bool)
        mask[4:21, 7:19] = True
        mask[100:118, 230:267] = True
        old_labels, old_count = connected_person_components(mask)
        new_labels, new_count = _connected_person_components_array(mask)
        self.assertEqual(old_count, new_count)
        old_array = old_labels.numpy()
        self.assertEqual(old_array.shape, new_labels.shape)
        self.assertEqual(new_labels.dtype, np.int32)
        # Component numbers are private; foreground and pairwise membership
        # are the scientific invariants used by the p025 association policy.
        self.assertTrue(np.array_equal(old_array > 0, new_labels > 0))
        points = ((5, 8), (18, 10), (103, 240), (0, 0))
        for left in points:
            for right in points:
                self.assertEqual(
                    old_array[left] == old_array[right],
                    new_labels[left] == new_labels[right],
                )

    def test_p025_v3_is_bit_identical_to_qualified_v1(self) -> None:
        generator = torch.Generator().manual_seed(20260910)
        for _iteration in range(20):
            count = 20 + int(torch.randint(0, 61, (), generator=generator))
            labels = torch.randint(0, 2, (count,), generator=generator)
            scores = torch.rand((count,), generator=generator, dtype=torch.float32)
            left_top = torch.rand((count, 2), generator=generator) * torch.tensor(
                [700.0, 390.0]
            )
            extent = 2.0 + torch.rand((count, 2), generator=generator) * 60.0
            detections = {
                "boxes": torch.cat((left_top, left_top + extent), dim=1).float(),
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
                (1, 3, CONTENT_H, CONTENT_W),
                generator=generator,
                dtype=torch.float32,
            )
            expected, expected_indices = apply_p025_service_policy_optimized(
                {"semantic_logits": logits}, detections
            )
            components, _count = _connected_person_components_array(
                person_mask_from_logits(logits)
            )
            observed, observed_indices = apply_p025_service_policy_v3(
                detections, components
            )
            self.assertTrue(tree_bitwise_equal(expected, observed))
            self.assertTrue(torch.equal(expected_indices, observed_indices))


@unittest.skipUnless(torch.cuda.is_available(), "deferred validation is CUDA only")
class DeferredFiniteValidatorTests(unittest.TestCase):
    """The deferred verdict must describe the checked output, not recycled memory."""

    DEVICE = "cuda:0"
    ELEMENTS = 1 << 22

    def test_verdict_survives_allocator_recycling_of_checked_storage(self) -> None:
        # The validation stream reads compute-stream storage. If that storage is
        # released before the isfinite kernel runs, the allocator may hand the
        # block to the next compute-stream allocation. Every checked tensor here
        # is all-zero, so a non-finite verdict can only describe recycled memory.
        device = torch.device(self.DEVICE)
        validator = _DeferredFiniteValidator(device)
        for _ in range(40):
            checked = torch.zeros(self.ELEMENTS, device=device, dtype=torch.float32)
            wide = torch.zeros(
                (64, self.ELEMENTS // 64), device=device, dtype=torch.float32
            )
            validator.launch({"checked": checked, "wide": wide}, "recycling probe")
            del checked, wide
            poison = [
                torch.full(
                    (self.ELEMENTS,), float("nan"), device=device, dtype=torch.float32
                )
                for _ in range(2)
            ]
            validator.resolve()
            del poison
        self.assertEqual(validator.asynchronous_verdict_corrections, 0)

    def test_injected_nonfinite_still_stops_the_frame(self) -> None:
        device = torch.device(self.DEVICE)
        validator = _DeferredFiniteValidator(device)
        validator.launch(
            {"injected": torch.tensor([float("nan")], device=device)},
            "injected fault",
        )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            validator.resolve()

    def test_failed_predicate_still_stops_the_frame(self) -> None:
        device = torch.device(self.DEVICE)
        validator = _DeferredFiniteValidator(device)
        validator.launch_predicate(
            (torch.tensor([-1.0], device=device) > 0).all(), "non-positive dimension"
        )
        with self.assertRaisesRegex(RuntimeError, "non-positive dimension"):
            validator.resolve()

    def test_discard_releases_an_abandoned_frame(self) -> None:
        device = torch.device(self.DEVICE)
        validator = _DeferredFiniteValidator(device)
        validator.launch({"a": torch.zeros(4, device=device)}, "abandoned")
        validator.discard()
        validator.reset()


if __name__ == "__main__":
    unittest.main()

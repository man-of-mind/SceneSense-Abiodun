"""CPU-only contract tests for the detached serialization boundary."""

from __future__ import annotations

import unittest
import threading
from types import MappingProxyType, SimpleNamespace

import torch

from rl_agent.splitfusion_live_dispatch_v1.frame_context import FrameContextV1, Pose6D

from .detached_tail import DetachedOptimizedTailAdapter, DetachedTailWorkProduct


FIELDS = ("sample_id", "frame_id", "score", "prediction_index")


class _Infer:
    FIELDS = FIELDS

    @staticmethod
    def record(detections: dict[str, torch.Tensor], row: dict[str, object], index: int) -> dict[str, object]:
        return {
            "sample_id": row["sample_id"],
            "frame_id": row["frame_id"],
            "score": float(detections["scores"][index]),
        }


class _Ledger:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def bump(self, name: str) -> None:
        self.calls.append(name)


def work() -> DetachedTailWorkProduct:
    context = FrameContextV1(
        stream_id="stream",
        frame_id=7,
        sequence_id=3,
        capture_timestamp_ns=123,
        ego_world=Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        camera_model_sha256="a" * 64,
        camera_mount_sha256="b" * 64,
    )
    perception = {
        "scores": torch.tensor([0.75], dtype=torch.float32),
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    }
    return DetachedTailWorkProduct(
        perception=MappingProxyType(perception),
        perception_cpu=MappingProxyType(perception),
        original_indices=torch.tensor([4]),
        original_indices_cpu=torch.tensor([4]),
        semantic_logits=torch.zeros((1, 3, 2, 2)),
        semantic_labels=torch.zeros((2, 2), dtype=torch.long),
        outputs=MappingProxyType({}),
        camera_world=torch.eye(4, dtype=torch.float64),
        frame_context=context,
        camera_pose_reconstruct_ns=1,
        output_tensor_count=2,
    )


class DetachedTailContractTest(unittest.TestCase):
    def test_serializer_is_product_scoped_and_cpu_only(self) -> None:
        adapter = object.__new__(DetachedOptimizedTailAdapter)
        adapter._publication_call_lock = threading.Lock()
        adapter._ledger = _Ledger()
        adapter._base = SimpleNamespace(infer=_Infer())
        adapter._record_fields = FIELDS
        serialized = adapter.serialize_product(work())
        self.assertEqual(serialized.record_count, 1)
        self.assertIn(b'"frame_id":7', serialized.serialized_records)
        self.assertIn(b'"prediction_index":4', serialized.serialized_records)
        self.assertEqual(adapter._ledger.calls, ["service_record_serialization"])
        snapshot = adapter.snapshot(serialized)
        self.assertEqual(snapshot.frame_context.frame_id, 7)
        self.assertEqual(snapshot.serialized_records, serialized.serialized_records)

    def test_non_cpu_publication_tensor_is_rejected(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        value = work()
        gpu_scores = value.perception_cpu["scores"].cuda()
        invalid = DetachedTailWorkProduct(
            **{
                **value.__dict__,
                "perception_cpu": MappingProxyType(
                    {**value.perception_cpu, "scores": gpu_scores}
                ),
            }
        )
        adapter = object.__new__(DetachedOptimizedTailAdapter)
        adapter._publication_call_lock = threading.Lock()
        adapter._ledger = _Ledger()
        adapter._base = SimpleNamespace(infer=_Infer())
        adapter._record_fields = FIELDS
        with self.assertRaisesRegex(Exception, "non-CPU"):
            adapter.serialize_product(invalid)


if __name__ == "__main__":
    unittest.main()

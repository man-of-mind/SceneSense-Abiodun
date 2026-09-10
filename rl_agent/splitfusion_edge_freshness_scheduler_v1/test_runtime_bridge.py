from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from rl_agent.splitfusion_edge_freshness_scheduler_v1.pipeline import (
    CandidatePolicy,
    PipelineConfig,
    PipelineWorkerError,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.runtime_bridge import (
    EdgeFrameRequest,
    PipelinedSplitEdgeBridge,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.scheduler import (
    FrameTicket,
    TerminalReason,
)


def make_ticket(sequence: int, wire_bytes: bytes = b"SFD1") -> FrameTicket:
    now = time.time_ns()
    return FrameTicket(
        run_id="run",
        cell_id="cell",
        stream_id="ue-1",
        frame_id=500 + sequence,
        sequence_id=sequence,
        action_id=71,
        capture_timestamp_ns=now,
        edge_arrival_timestamp_ns=now,
        feature_bytes=len(wire_bytes),
    )


class FakeRuntime:
    def __init__(self, *, drift: str | None = None) -> None:
        self.drift = drift

    def process(self, wire_bytes: bytes, *, transmitted_action_id: int) -> object:
        ticket = self.ticket
        context = SimpleNamespace(
            stream_id=ticket.stream_id,
            frame_id=ticket.frame_id + int(self.drift == "frame"),
        )
        metadata = SimpleNamespace(
            action_id=transmitted_action_id,
            sequence_id=ticket.sequence_id,
            capture_timestamp_ns=ticket.capture_timestamp_ns,
            frame_context=context,
        )
        return SimpleNamespace(
            metadata=metadata,
            total_received_bytes=len(wire_bytes),
            serialized_output=b"[]",
        )


class RuntimeBridgeTest(unittest.TestCase):
    def test_strict_identity_and_byte_bound_result_reaches_publisher(self) -> None:
        runtime = FakeRuntime()
        published: list[tuple[int, bytes]] = []
        bridge = PipelinedSplitEdgeBridge(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            edge_runtime=runtime,
            publish_result=lambda ticket, result: published.append(
                (ticket.sequence_id, result.serialized_output)
            ),
        )
        frame = make_ticket(1)
        runtime.ticket = frame
        bridge.start()
        bridge.offer(frame, EdgeFrameRequest(b"SFD1", 71))
        outcomes = bridge.close_and_join(timeout_s=2.0)
        self.assertEqual(published, [(1, b"[]")])
        self.assertEqual(outcomes[0].reason, TerminalReason.MAP_INSTALLED)
        self.assertEqual(bridge.snapshot().compute_completed, 1)

    def test_context_identity_drift_fails_closed(self) -> None:
        runtime = FakeRuntime(drift="frame")
        bridge = PipelinedSplitEdgeBridge(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            edge_runtime=runtime,
            publish_result=lambda _ticket, _result: None,
        )
        frame = make_ticket(1)
        runtime.ticket = frame
        bridge.start()
        bridge.offer(frame, EdgeFrameRequest(b"SFD1", 71))
        with self.assertRaises(PipelineWorkerError):
            bridge.close_and_join(timeout_s=2.0)
        self.assertEqual(bridge.outcomes[0].reason, TerminalReason.PROCESSING_FAILED)

    def test_request_action_and_bytes_are_reconciled_before_runtime(self) -> None:
        runtime = FakeRuntime()
        bridge = PipelinedSplitEdgeBridge(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            edge_runtime=runtime,
            publish_result=lambda _ticket, _result: None,
        )
        frame = make_ticket(1)
        runtime.ticket = frame
        bridge.start()
        bridge.offer(frame, EdgeFrameRequest(b"SFD1x", 71))
        with self.assertRaises(PipelineWorkerError):
            bridge.close_and_join(timeout_s=2.0)
        self.assertEqual(bridge.outcomes[0].reason, TerminalReason.PROCESSING_FAILED)


if __name__ == "__main__":
    unittest.main()

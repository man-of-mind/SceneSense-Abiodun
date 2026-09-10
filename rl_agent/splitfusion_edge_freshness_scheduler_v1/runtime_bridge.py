"""Strict bridge from SFD1 edge dispatch to the bounded pipeline candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .pipeline import BoundedTwoStagePipeline, PipelineConfig, PipelineSnapshot
from .scheduler import Admission, FrameTicket, TerminalFeedback


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EdgeFrameRequest:
    wire_bytes: bytes
    transmitted_action_id: int

    def __post_init__(self) -> None:
        _require(isinstance(self.wire_bytes, bytes), "wire_bytes must be immutable bytes")
        _require(self.transmitted_action_id >= 0, "action ID is invalid")


class PipelinedSplitEdgeBridge:
    """Run the frozen edge on its sole owner and publish on a second owner.

    The current ``PreloadedSplitEdgeRuntime.process`` call still owns decode,
    tail, post-processing, and serialization.  Only its already-serialized
    result crosses to the publication callback.  This is the largest safe
    concurrency boundary before the tail's singleton snapshot is refactored
    and independently qualified for output parity.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig,
        edge_runtime: Any,
        publish_result: Callable[[FrameTicket, Any], None],
        feedback_sink: Callable[[TerminalFeedback], None] | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        _require(callable(getattr(edge_runtime, "process", None)), "edge runtime is invalid")
        _require(callable(publish_result), "result publisher is required")
        self._runtime = edge_runtime
        self._publish_result = publish_result
        arguments: dict[str, Any] = {
            "config": config,
            "compute": self._process,
            "publish": self._publish,
            "feedback_sink": feedback_sink,
        }
        if clock_ns is not None:
            arguments["clock_ns"] = clock_ns
        self._pipeline = BoundedTwoStagePipeline(**arguments)

    def _process(self, ticket: FrameTicket, request: EdgeFrameRequest) -> Any:
        _require(
            request.transmitted_action_id == ticket.action_id,
            "ticket/request action identity mismatch",
        )
        _require(
            len(request.wire_bytes) == ticket.feature_bytes,
            "ticket/request byte accounting mismatch",
        )
        result = self._runtime.process(
            request.wire_bytes,
            transmitted_action_id=request.transmitted_action_id,
        )
        metadata = getattr(result, "metadata", None)
        _require(metadata is not None, "edge result lacks dispatch metadata")
        _require(metadata.action_id == ticket.action_id, "result action identity drift")
        _require(
            metadata.sequence_id == ticket.sequence_id,
            "result sequence identity drift",
        )
        _require(
            metadata.capture_timestamp_ns == ticket.capture_timestamp_ns,
            "result capture timestamp drift",
        )
        context = getattr(metadata, "frame_context", None)
        _require(context is not None, "edge result lacks SFD1 frame context")
        _require(context.stream_id == ticket.stream_id, "result stream identity drift")
        _require(context.frame_id == ticket.frame_id, "result frame identity drift")
        _require(
            getattr(result, "total_received_bytes", None) == ticket.feature_bytes,
            "edge result byte accounting drift",
        )
        _require(
            getattr(result, "serialized_output", None) is not None,
            "edge result lacks serialized output",
        )
        return result

    def _publish(self, ticket: FrameTicket, result: Any) -> None:
        self._publish_result(ticket, result)

    def start(self) -> None:
        self._pipeline.start()

    def offer(
        self,
        ticket: FrameTicket,
        request: EdgeFrameRequest,
        *,
        now_ns: int | None = None,
    ) -> Admission:
        return self._pipeline.offer(ticket, request, now_ns=now_ns)

    def close_and_join(self, *, timeout_s: float = 30.0) -> tuple[TerminalFeedback, ...]:
        return self._pipeline.close_and_join(timeout_s=timeout_s)

    def snapshot(self) -> PipelineSnapshot:
        return self._pipeline.snapshot()

    @property
    def outcomes(self) -> tuple[TerminalFeedback, ...]:
        return self._pipeline.outcomes

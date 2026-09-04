"""Unaggregated monotonic-nanosecond stage boundaries for later qualification."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


UE_STAGES = (
    "total_ue_preparation",
    "front_backbone",
    "ranker_selection",
    "ae_encode",
    "quantize_pack",
    "zstd_compression",
)
EDGE_STAGES = (
    "total_edge_processing",
    "zstd_decompression",
    "unpack_dequantize",
    "ae_decode",
    "frozen_tail",
    "output_serialization",
)


@dataclass(frozen=True)
class StageBoundary:
    name: str
    started_monotonic_ns: int
    finished_monotonic_ns: int


@dataclass(frozen=True)
class TimingTrace:
    clock: str
    boundaries: tuple[StageBoundary, ...]
    latency_published: bool = False


class StageRecorder:
    """Record raw boundaries only; deliberately provides no latency summary."""

    def __init__(self, allowed_stages: tuple[str, ...]) -> None:
        self._allowed = frozenset(allowed_stages)
        self._boundaries: list[StageBoundary] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self._allowed:
            raise ValueError(f"unregistered timing stage: {name}")
        started = time.monotonic_ns()
        try:
            yield
        finally:
            self._boundaries.append(
                StageBoundary(
                    name=name,
                    started_monotonic_ns=started,
                    finished_monotonic_ns=time.monotonic_ns(),
                )
            )

    def snapshot(self) -> TimingTrace:
        return TimingTrace(clock="time.monotonic_ns", boundaries=tuple(self._boundaries))

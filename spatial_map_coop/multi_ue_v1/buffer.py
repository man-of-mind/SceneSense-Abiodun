"""Bounded causal buffering for validated multi-UE observation batches."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .contracts import MultiUEContractError, ObservationBatch


@dataclass(frozen=True)
class IngestResult:
    disposition: str
    source_key: tuple[str, str, str]
    frame_id: int
    retained_frames: int


@dataclass(frozen=True)
class SourceFrame:
    batch: ObservationBatch
    age_ns: int


@dataclass(frozen=True)
class MultiUESnapshot:
    clock_domain: str
    snapshot_timestamp_ns: int
    sources: tuple[SourceFrame, ...]
    capture_spread_ns: int
    alignment_tolerance_ns: int
    aligned_for_fusion: bool

    @property
    def raw_object_count(self) -> int:
        return sum(len(source.batch.objects) for source in self.sources)


class MultiUEFrameBuffer:
    """Retain a bounded history and select one causal frame per UE stream.

    This is a raw observation store, not an association or tracking engine.
    A snapshot explicitly reports whether selected source frames fall within
    the requested alignment tolerance; downstream fusion must refuse an
    unaligned snapshot rather than silently average asynchronous objects.
    """

    def __init__(self, *, max_frames_per_source: int = 32) -> None:
        if isinstance(max_frames_per_source, bool) or max_frames_per_source < 1:
            raise ValueError("max_frames_per_source must be a positive integer")
        self._maximum = int(max_frames_per_source)
        self._frames: dict[tuple[str, str, str], list[ObservationBatch]] = {}
        self._identities: dict[tuple[str, str, str, int], str] = {}
        self._lock = RLock()

    def ingest(self, batch: ObservationBatch) -> IngestResult:
        if not isinstance(batch, ObservationBatch):
            raise TypeError("batch must be ObservationBatch")
        with self._lock:
            prior_hash = self._identities.get(batch.identity)
            if prior_hash is not None:
                if prior_hash != batch.source_payload_sha256:
                    raise MultiUEContractError(
                        "conflicting payload for an existing UE/session/stream/frame identity"
                    )
                retained = len(self._frames[batch.source_key])
                return IngestResult("DUPLICATE_IDENTICAL", batch.source_key, batch.frame_id, retained)

            frames = self._frames.setdefault(batch.source_key, [])
            if (
                len(frames) == self._maximum
                and (batch.capture_timestamp_ns, batch.frame_id)
                <= (frames[0].capture_timestamp_ns, frames[0].frame_id)
            ):
                return IngestResult(
                    "STALE_NOT_RETAINED", batch.source_key, batch.frame_id, len(frames)
                )
            frames.append(batch)
            frames.sort(key=lambda item: (item.capture_timestamp_ns, item.frame_id))
            self._identities[batch.identity] = batch.source_payload_sha256
            while len(frames) > self._maximum:
                removed = frames.pop(0)
                self._identities.pop(removed.identity, None)
            return IngestResult("ACCEPTED", batch.source_key, batch.frame_id, len(frames))

    def snapshot(
        self,
        *,
        clock_domain: str,
        snapshot_timestamp_ns: int,
        maximum_source_age_ns: int,
        alignment_tolerance_ns: int,
    ) -> MultiUESnapshot:
        if snapshot_timestamp_ns < 0:
            raise ValueError("snapshot_timestamp_ns must be nonnegative")
        if maximum_source_age_ns < 0 or alignment_tolerance_ns < 0:
            raise ValueError("age and alignment bounds must be nonnegative")
        selected: list[SourceFrame] = []
        with self._lock:
            for source_key in sorted(self._frames):
                candidates = self._frames[source_key]
                chosen = None
                for candidate in reversed(candidates):
                    if (
                        candidate.clock_domain == clock_domain
                        and candidate.capture_timestamp_ns <= snapshot_timestamp_ns
                    ):
                        chosen = candidate
                        break
                if chosen is None:
                    continue
                age_ns = snapshot_timestamp_ns - chosen.capture_timestamp_ns
                if age_ns <= maximum_source_age_ns:
                    selected.append(SourceFrame(chosen, age_ns))

        capture_times = [source.batch.capture_timestamp_ns for source in selected]
        spread_ns = max(capture_times) - min(capture_times) if capture_times else 0
        return MultiUESnapshot(
            clock_domain=clock_domain,
            snapshot_timestamp_ns=snapshot_timestamp_ns,
            sources=tuple(selected),
            capture_spread_ns=spread_ns,
            alignment_tolerance_ns=alignment_tolerance_ns,
            aligned_for_fusion=(len(selected) <= 1 or spread_ns <= alignment_tolerance_ns),
        )

    def source_count(self) -> int:
        with self._lock:
            return len(self._frames)

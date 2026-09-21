"""Honest, bounded live metrics for the cooperative-map viewer.

Naming rules enforced here, not left to a template author:

* Every latency has an **explicit boundary label** naming both endpoints. There
  is no metric called "network latency" and no residual is given that name.
* A quantity that would subtract two different clocks returns the literal string
  ``N/A - incompatible clock domains``. It never returns a number.
* An unavailable field returns ``"N/A"``. It never returns ``0.0``.
* Percentile windows are bounded and deterministic: a fixed-capacity FIFO with
  an explicit, documented percentile rule, so the same inputs always give the
  same output and no full-history scan ever happens.

Nothing in this module reads or writes association state. The metrics layer is
a consumer of already-computed snapshots; ``MetricsRegistry.observe_*`` accepts
plain numbers and strings only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
from typing import Any, Deque, Mapping, Sequence

from .normalized import (
    CLOCK_UNIX_WALL_NS,
    CLOCK_UNIX_WALL_S,
    ClockStamp,
    EdgeTimingProvenance,
    NormalizedCoopBatch,
    clocks_compatible,
)


NOT_AVAILABLE = "N/A"
INCOMPATIBLE_CLOCKS = "N/A - incompatible clock domains"

# Default rolling window: the last 120 installed updates, as registered.
DEFAULT_WINDOW = 120

# Boundary labels. Each names its two endpoints so a reader never has to guess
# which interval is plotted.
LABEL_TAIL_TO_INSTALL = "edge tail-complete -> map-install"
LABEL_PUBLISH_TO_INSTALL = "edge publish-start -> map-install"
LABEL_CAPTURE_TO_INSTALL = "capture -> map-install AoI"
LABEL_DEMO_POST = "demo-control delivery time (HTTP POST, localhost)"
LABEL_LOCALHOST_MODEL = "localhost model path"
LABEL_POLL_RTT = "browser poll round-trip (fetch issue -> response parsed)"
LABEL_SNAPSHOT_BUILD = "map snapshot build (server, lock-to-JSON)"

# Labels that must never be attached to any metric in this module. Kept as data
# so a test can assert the prohibition instead of trusting review.
# Must all be lowercase: the guard lowercases the candidate label before
# comparing, so a mixed-case entry here would silently never match.
FORBIDDEN_LABELS = (
    "network latency",
    "5g latency",
    "radio latency",
    "uplink latency",
    "end-to-end network",
    "wireless latency",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Deterministic nearest-rank percentile on a sorted copy.

    Nearest-rank (not interpolated) so the reported value is always an actually
    observed sample, and so the result is exactly reproducible.
    """

    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty window")
    rank = max(1, math.ceil(fraction * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


@dataclass(frozen=True)
class WindowSummary:
    label: str
    unit: str
    count: int
    capacity: int
    current: float | str
    p50: float | str
    p95: float | str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "unit": self.unit,
            "count": self.count,
            "capacity": self.capacity,
            "current": self.current,
            "p50": self.p50,
            "p95": self.p95,
        }


class RollingWindow:
    """Bounded FIFO of finite samples with deterministic percentiles."""

    def __init__(self, label: str, *, unit: str = "ms", capacity: int = DEFAULT_WINDOW) -> None:
        if int(capacity) <= 0:
            raise ValueError("rolling window capacity must be positive")
        lowered = str(label).lower()
        for forbidden in FORBIDDEN_LABELS:
            if forbidden in lowered:
                raise ValueError(
                    f"metric label {label!r} uses the prohibited term {forbidden!r}; "
                    "every latency must name its exact boundary"
                )
        self.label = str(label)
        self.unit = str(unit)
        self.capacity = int(capacity)
        self._samples: Deque[float] = deque(maxlen=self.capacity)

    def observe(self, value: Any) -> bool:
        """Append a finite sample. Non-finite/absent input is dropped, not zeroed."""

        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        self._samples.append(numeric)
        return True

    def summary(self) -> WindowSummary:
        if not self._samples:
            return WindowSummary(
                self.label, self.unit, 0, self.capacity,
                NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE,
            )
        values = list(self._samples)
        return WindowSummary(
            self.label,
            self.unit,
            len(values),
            self.capacity,
            float(values[-1]),
            _percentile(values, 0.50),
            _percentile(values, 0.95),
        )

    def samples(self) -> tuple[float, ...]:
        return tuple(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


def cross_clock_age_ms(
    capture: ClockStamp | None, reference: ClockStamp | None
) -> float | str:
    """Age between two stamps, or the incompatible-clock string.

    This is the single place the CARLA-simulation/wall-clock question is decided.
    """

    if capture is None or reference is None:
        return NOT_AVAILABLE
    if not clocks_compatible(capture, reference):
        return INCOMPATIBLE_CLOCKS
    return (reference.timestamp_ns - capture.timestamp_ns) / 1_000_000.0


@dataclass(frozen=True)
class DirectTiming:
    """Direct-path stage latencies derived only from proven-compatible stamps."""

    tail_complete_to_install_ms: float | str = NOT_AVAILABLE
    publish_start_to_install_ms: float | str = NOT_AVAILABLE
    capture_to_install_aoi_ms: float | str = NOT_AVAILABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            LABEL_TAIL_TO_INSTALL: self.tail_complete_to_install_ms,
            LABEL_PUBLISH_TO_INSTALL: self.publish_start_to_install_ms,
            LABEL_CAPTURE_TO_INSTALL: self.capture_to_install_aoi_ms,
        }


def direct_timing_for(batch: NormalizedCoopBatch) -> DirectTiming:
    """Compute direct-path stage latencies, or N/A for every unprovable one.

    Requirements enforced:
      * an install timestamp must exist (a terminal that installed nothing has
        no install latency);
      * the edge stage stamp must be present and finite;
      * the edge timing clock and the install clock must be the same clock;
      * the interval must be non-negative (a boundary inversion is excluded, not
        clamped to zero).
    """

    terminal = batch.map_terminal
    install = terminal.install_timestamp
    if install is None or not terminal.available:
        return DirectTiming()

    timing: EdgeTimingProvenance = batch.edge_timing
    install_seconds = install.timestamp_ns / 1_000_000_000.0

    def stage_ms(name: str) -> float | str:
        if not timing.available:
            return NOT_AVAILABLE
        # Production edge_timing stamps are unix wall seconds; the install stamp
        # is unix wall ns. Same clock, different unit -> comparable.
        if timing.clock_domain != CLOCK_UNIX_WALL_S or install.clock_domain != CLOCK_UNIX_WALL_NS:
            return INCOMPATIBLE_CLOCKS
        stamp = timing.get(name)
        if stamp is None:
            return NOT_AVAILABLE
        delta_ms = (install_seconds - float(stamp)) * 1000.0
        if not math.isfinite(delta_ms) or delta_ms < 0.0:
            return NOT_AVAILABLE
        return delta_ms

    aoi: float | str = NOT_AVAILABLE
    if batch.capture.clock_domain != install.clock_domain:
        aoi = INCOMPATIBLE_CLOCKS
    else:
        candidate = (install.timestamp_ns - batch.capture.timestamp_ns) / 1_000_000.0
        aoi = candidate if math.isfinite(candidate) and candidate >= 0.0 else NOT_AVAILABLE

    return DirectTiming(
        tail_complete_to_install_ms=stage_ms("tail_complete_wall_s"),
        publish_start_to_install_ms=stage_ms("publish_start_wall_s"),
        capture_to_install_aoi_ms=aoi,
    )


class StreamRateMeter:
    """Per-stream update rate and last-accepted frame, from receipt stamps only.

    The rate is computed from monotonic wall receipts, which is legitimate for
    both transports: it measures how often the map *received* an update and makes
    no claim about capture-to-install age.
    """

    def __init__(self, *, capacity: int = 32) -> None:
        self.capacity = int(capacity)
        self._receipts: dict[str, Deque[float]] = {}
        self._last_frame: dict[str, int] = {}
        self._accepted: dict[str, int] = {}

    def observe(self, stream_id: str, *, frame_id: int, received_wall_s: float) -> None:
        if not math.isfinite(float(received_wall_s)):
            return
        window = self._receipts.setdefault(stream_id, deque(maxlen=self.capacity))
        window.append(float(received_wall_s))
        self._last_frame[stream_id] = int(frame_id)
        self._accepted[stream_id] = self._accepted.get(stream_id, 0) + 1

    def report(self, *, now_wall_s: float) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for stream_id, window in sorted(self._receipts.items()):
            rate: float | str = NOT_AVAILABLE
            if len(window) >= 2:
                span = window[-1] - window[0]
                rate = (len(window) - 1) / span if span > 0.0 else NOT_AVAILABLE
            since: float | str = NOT_AVAILABLE
            if window:
                candidate = (float(now_wall_s) - window[-1]) * 1000.0
                since = candidate if math.isfinite(candidate) else NOT_AVAILABLE
            result[stream_id] = {
                "update_rate_hz": rate,
                "last_accepted_frame_id": self._last_frame.get(stream_id, NOT_AVAILABLE),
                "since_last_received_ms": since,
                "accepted_updates": self._accepted.get(stream_id, 0),
                "window_samples": len(window),
            }
        return result


def association_reduction(raw_observations: int, associated_representations: int) -> float | str:
    """``1 - associated/raw``. N/A when there are no raw observations."""

    raw = int(raw_observations)
    if raw <= 0:
        return NOT_AVAILABLE
    return 1.0 - (int(associated_representations) / raw)


class MetricsRegistry:
    """Thread-safe container for the viewer's rolling windows and counters.

    It holds numbers and labels only. It never touches tracks, observations or
    association state, so it cannot perturb the scientific path.
    """

    def __init__(self, *, capacity: int = DEFAULT_WINDOW) -> None:
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self.tail_to_install = RollingWindow(LABEL_TAIL_TO_INSTALL, capacity=self.capacity)
        self.publish_to_install = RollingWindow(
            LABEL_PUBLISH_TO_INSTALL, capacity=self.capacity
        )
        self.capture_to_install = RollingWindow(
            LABEL_CAPTURE_TO_INSTALL, capacity=self.capacity
        )
        self.snapshot_build = RollingWindow(LABEL_SNAPSHOT_BUILD, capacity=self.capacity)
        self.demo_post = RollingWindow(LABEL_DEMO_POST, capacity=self.capacity)
        self.rates = StreamRateMeter()
        self.outcomes: dict[str, int] = {}

    def observe_direct_batch(self, batch: NormalizedCoopBatch) -> DirectTiming:
        timing = direct_timing_for(batch)
        with self._lock:
            self.tail_to_install.observe(timing.tail_complete_to_install_ms)
            self.publish_to_install.observe(timing.publish_start_to_install_ms)
            self.capture_to_install.observe(timing.capture_to_install_aoi_ms)
            outcome = batch.map_terminal.outcome
            if outcome is not None:
                self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        return timing

    def observe_snapshot_build_ms(self, value: Any) -> None:
        with self._lock:
            self.snapshot_build.observe(value)

    def observe_demo_post_ms(self, value: Any) -> None:
        with self._lock:
            self.demo_post.observe(value)

    def observe_receipt(self, stream_id: str, *, frame_id: int, received_wall_s: float) -> None:
        with self._lock:
            self.rates.observe(
                stream_id, frame_id=frame_id, received_wall_s=received_wall_s
            )

    def report(self, *, now_wall_s: float) -> dict[str, Any]:
        with self._lock:
            return {
                "window_capacity": self.capacity,
                "latency_windows": [
                    self.tail_to_install.summary().as_dict(),
                    self.publish_to_install.summary().as_dict(),
                    self.capture_to_install.summary().as_dict(),
                ],
                "service_windows": [
                    self.snapshot_build.summary().as_dict(),
                    self.demo_post.summary().as_dict(),
                ],
                "streams": self.rates.report(now_wall_s=now_wall_s),
                "terminal_outcome_counts": dict(sorted(self.outcomes.items())),
            }


def demo_transport_labels() -> dict[str, str]:
    """The only labels the HTTP demo transport is allowed to display."""

    return {
        "cross_clock_wall_aoi": INCOMPATIBLE_CLOCKS,
        "http_post_duration": LABEL_DEMO_POST,
        "model_execution": LABEL_LOCALHOST_MODEL,
        "radio_claim": "none - this transport carries no radio",
    }

"""Deterministic single-edge-worker freshness scheduling simulator."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class QueuePolicy(str, Enum):
    FIFO = "FIFO"
    LATEST_ONLY = "LATEST_ONLY"


class SimulationReason(str, Enum):
    INSTALLED = "INSTALLED"
    SUPERSEDED_PENDING = "SUPERSEDED_PENDING"
    QUEUE_WAIT_BUDGET_EXCEEDED = "QUEUE_WAIT_BUDGET_EXCEEDED"
    PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL = (
        "PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL"
    )
    PROCESSING_HORIZON_EXPIRED_BEFORE_SERVICE = (
        "PROCESSING_HORIZON_EXPIRED_BEFORE_SERVICE"
    )
    PROCESSING_HORIZON_EXPIRED_BEFORE_INSTALL = (
        "PROCESSING_HORIZON_EXPIRED_BEFORE_INSTALL"
    )
    OUT_OF_ORDER_ARRIVAL = "OUT_OF_ORDER_ARRIVAL"


@dataclass(frozen=True)
class SimulationFrame:
    frame_id: int
    sequence_id: int
    capture_ns: int
    arrival_ns: int
    service_ns: int
    feature_bytes: int
    arrival_observed: bool = True
    service_observed: bool = True

    def __post_init__(self) -> None:
        _require(self.frame_id >= 0, "frame_id must be non-negative")
        _require(self.sequence_id >= 0, "sequence_id must be non-negative")
        _require(self.capture_ns >= 0, "capture_ns must be non-negative")
        _require(self.arrival_ns >= self.capture_ns, "arrival precedes capture")
        _require(self.service_ns > 0, "service_ns must be positive")
        _require(self.feature_bytes >= 0, "feature_bytes must be non-negative")


@dataclass(frozen=True)
class SimulationOutcome:
    frame: SimulationFrame
    reason: SimulationReason
    terminal_ns: int
    service_start_ns: int | None = None

    @property
    def queue_wait_ns(self) -> int | None:
        if self.service_start_ns is None:
            return None
        return self.service_start_ns - self.frame.arrival_ns

    @property
    def install_aoi_ns(self) -> int | None:
        if self.reason is not SimulationReason.INSTALLED:
            return None
        return self.terminal_ns - self.frame.capture_ns


@dataclass(frozen=True)
class SimulationConfig:
    queue_policy: QueuePolicy
    queue_wait_budget_ns: int | None
    processing_horizon_ns: int | None
    service_target_ns: int = 100_000_000

    def __post_init__(self) -> None:
        if self.queue_wait_budget_ns is not None:
            _require(self.queue_wait_budget_ns >= 0, "queue budget is invalid")
        if self.processing_horizon_ns is not None:
            _require(self.processing_horizon_ns > 0, "horizon is invalid")
        _require(self.service_target_ns > 0, "service target is invalid")


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    outcomes: tuple[SimulationOutcome, ...]
    observation_start_ns: int
    observation_end_ns: int

    def summary(self) -> dict[str, Any]:
        by_reason = {
            reason.value: sum(outcome.reason is reason for outcome in self.outcomes)
            for reason in SimulationReason
        }
        installed = sorted(
            (
                outcome
                for outcome in self.outcomes
                if outcome.reason is SimulationReason.INSTALLED
            ),
            key=lambda outcome: outcome.terminal_ns,
        )
        install_aoi_ms = [
            float(outcome.install_aoi_ns) / 1_000_000.0
            for outcome in installed
            if outcome.install_aoi_ns is not None
        ]
        queue_wait_ms = [
            float(outcome.queue_wait_ns) / 1_000_000.0
            for outcome in self.outcomes
            if outcome.queue_wait_ns is not None
        ]
        duration_s = max(
            0.0, (self.observation_end_ns - self.observation_start_ns) / 1e9
        )
        aoi = _time_weighted_aoi(
            installed,
            start_ns=self.observation_start_ns,
            end_ns=self.observation_end_ns,
            threshold_ns=self.config.service_target_ns,
        )
        total = len(self.outcomes)
        return {
            "queue_policy": self.config.queue_policy.value,
            "queue_wait_budget_ms": (
                None
                if self.config.queue_wait_budget_ns is None
                else self.config.queue_wait_budget_ns / 1_000_000.0
            ),
            "processing_horizon_ms": (
                None
                if self.config.processing_horizon_ns is None
                else self.config.processing_horizon_ns / 1_000_000.0
            ),
            "input_frames": total,
            "reason_counts": by_reason,
            "installed_frames": len(installed),
            "installed_fraction": None if not total else len(installed) / total,
            "installed_within_service_target": sum(
                value <= self.config.service_target_ns / 1_000_000.0
                for value in install_aoi_ms
            ),
            "install_aoi_ms_median": _percentile(install_aoi_ms, 0.5),
            "install_aoi_ms_p95": _percentile(install_aoi_ms, 0.95),
            "queue_wait_ms_median": _percentile(queue_wait_ms, 0.5),
            "queue_wait_ms_p95": _percentile(queue_wait_ms, 0.95),
            "observation_duration_s": duration_s,
            "installed_updates_per_s": (
                None if duration_s <= 0.0 else len(installed) / duration_s
            ),
            "modeled_feature_bytes": sum(
                outcome.frame.feature_bytes for outcome in self.outcomes
            ),
            "superseded_feature_bytes": sum(
                outcome.frame.feature_bytes
                for outcome in self.outcomes
                if outcome.reason is SimulationReason.SUPERSEDED_PENDING
            ),
            "expired_feature_bytes": sum(
                outcome.frame.feature_bytes
                for outcome in self.outcomes
                if outcome.reason
                in (
                    SimulationReason.QUEUE_WAIT_BUDGET_EXCEEDED,
                    SimulationReason.PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL,
                    SimulationReason.PROCESSING_HORIZON_EXPIRED_BEFORE_SERVICE,
                    SimulationReason.PROCESSING_HORIZON_EXPIRED_BEFORE_INSTALL,
                )
            ),
            **aoi,
            "observed_arrival_fraction": (
                None
                if not total
                else sum(outcome.frame.arrival_observed for outcome in self.outcomes)
                / total
            ),
            "observed_service_fraction": (
                None
                if not total
                else sum(outcome.frame.service_observed for outcome in self.outcomes)
                / total
            ),
        }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _time_weighted_aoi(
    installed: list[SimulationOutcome],
    *,
    start_ns: int,
    end_ns: int,
    threshold_ns: int,
) -> dict[str, float | None]:
    relevant = [outcome for outcome in installed if outcome.terminal_ns < end_ns]
    if not relevant:
        return {
            "time_weighted_map_aoi_ms": None,
            "map_time_above_service_target_fraction": None,
            "map_aoi_observation_duration_s": 0.0,
        }
    evaluation_start = max(start_ns, relevant[0].terminal_ns)
    if evaluation_start >= end_ns:
        return {
            "time_weighted_map_aoi_ms": None,
            "map_time_above_service_target_fraction": None,
            "map_aoi_observation_duration_s": 0.0,
        }
    area_ns2 = 0.0
    above_ns = 0
    for index, outcome in enumerate(relevant):
        interval_start = max(evaluation_start, outcome.terminal_ns)
        interval_end = (
            min(end_ns, relevant[index + 1].terminal_ns)
            if index + 1 < len(relevant)
            else end_ns
        )
        if interval_end <= interval_start:
            continue
        start_age = interval_start - outcome.frame.capture_ns
        duration = interval_end - interval_start
        area_ns2 += float(start_age) * duration + 0.5 * float(duration) ** 2
        crossing = outcome.frame.capture_ns + int(threshold_ns)
        above_ns += max(0, interval_end - max(interval_start, crossing))
    observation_ns = end_ns - evaluation_start
    return {
        "time_weighted_map_aoi_ms": area_ns2 / observation_ns / 1_000_000.0,
        "map_time_above_service_target_fraction": above_ns / observation_ns,
        "map_aoi_observation_duration_s": observation_ns / 1e9,
    }


def simulate(
    frames: Iterable[SimulationFrame], *, config: SimulationConfig
) -> SimulationResult:
    ordered = sorted(frames, key=lambda frame: (frame.arrival_ns, frame.sequence_id))
    _require(bool(ordered), "at least one frame is required")
    identities = {(frame.frame_id, frame.sequence_id) for frame in ordered}
    _require(len(identities) == len(ordered), "duplicate frame identity")

    pending: list[SimulationFrame] = []
    outcomes: dict[tuple[int, int], SimulationOutcome] = {}
    latest_sequence = -1
    active: SimulationFrame | None = None
    active_start_ns: int | None = None
    active_finish_ns: int | None = None
    index = 0
    now_ns = ordered[0].arrival_ns

    def terminal(
        frame: SimulationFrame,
        reason: SimulationReason,
        at_ns: int,
        *,
        service_start_ns: int | None = None,
    ) -> None:
        key = (frame.frame_id, frame.sequence_id)
        _require(key not in outcomes, "frame received multiple terminal outcomes")
        outcomes[key] = SimulationOutcome(
            frame=frame,
            reason=reason,
            terminal_ns=int(at_ns),
            service_start_ns=service_start_ns,
        )

    def admit(frame: SimulationFrame, at_ns: int) -> None:
        nonlocal latest_sequence
        if frame.sequence_id <= latest_sequence:
            terminal(frame, SimulationReason.OUT_OF_ORDER_ARRIVAL, at_ns)
            return
        latest_sequence = frame.sequence_id
        if (
            config.processing_horizon_ns is not None
            and at_ns - frame.capture_ns > config.processing_horizon_ns
        ):
            terminal(
                frame,
                SimulationReason.PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL,
                at_ns,
            )
            return
        if config.queue_policy is QueuePolicy.LATEST_ONLY:
            for displaced in pending:
                terminal(displaced, SimulationReason.SUPERSEDED_PENDING, at_ns)
            pending.clear()
        pending.append(frame)

    def start_next(at_ns: int) -> None:
        nonlocal active, active_start_ns, active_finish_ns
        while pending:
            frame = pending.pop(0)
            queue_wait = at_ns - frame.arrival_ns
            if (
                config.queue_wait_budget_ns is not None
                and queue_wait > config.queue_wait_budget_ns
            ):
                terminal(
                    frame,
                    SimulationReason.QUEUE_WAIT_BUDGET_EXCEEDED,
                    at_ns,
                )
                continue
            if (
                config.processing_horizon_ns is not None
                and at_ns - frame.capture_ns > config.processing_horizon_ns
            ):
                terminal(
                    frame,
                    SimulationReason.PROCESSING_HORIZON_EXPIRED_BEFORE_SERVICE,
                    at_ns,
                )
                continue
            active = frame
            active_start_ns = int(at_ns)
            active_finish_ns = int(at_ns) + frame.service_ns
            return
        active = None
        active_start_ns = None
        active_finish_ns = None

    while index < len(ordered) or active is not None or pending:
        next_arrival = ordered[index].arrival_ns if index < len(ordered) else None
        if active is not None and (
            next_arrival is None or int(active_finish_ns) <= next_arrival
        ):
            now_ns = int(active_finish_ns)
            if (
                config.processing_horizon_ns is not None
                and now_ns - active.capture_ns > config.processing_horizon_ns
            ):
                terminal(
                    active,
                    SimulationReason.PROCESSING_HORIZON_EXPIRED_BEFORE_INSTALL,
                    now_ns,
                    service_start_ns=active_start_ns,
                )
            else:
                terminal(
                    active,
                    SimulationReason.INSTALLED,
                    now_ns,
                    service_start_ns=active_start_ns,
                )
            active = None
            active_start_ns = None
            active_finish_ns = None
            start_next(now_ns)
            continue

        if next_arrival is not None:
            now_ns = next_arrival
            while index < len(ordered) and ordered[index].arrival_ns == now_ns:
                admit(ordered[index], now_ns)
                index += 1
            if active is None:
                start_next(now_ns)
            continue

        start_next(now_ns)

    _require(len(outcomes) == len(ordered), "terminal accounting is incomplete")
    return SimulationResult(
        config=config,
        outcomes=tuple(
            outcomes[(frame.frame_id, frame.sequence_id)]
            for frame in sorted(ordered, key=lambda item: item.sequence_id)
        ),
        observation_start_ns=min(frame.capture_ns for frame in ordered),
        observation_end_ns=max(frame.arrival_ns for frame in ordered),
    )

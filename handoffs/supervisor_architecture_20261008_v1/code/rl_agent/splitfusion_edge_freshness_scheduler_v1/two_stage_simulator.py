"""Deterministic counterfactual simulator for the detached two-stage edge.

The simulator models the scheduling semantics qualified by the live edge:
one non-preemptive compute owner, one non-preemptive publication owner, and a
depth-one latest pending slot in front of each.  It does not simulate radio
delivery; callers pass the measured complete-reassembly time or ``None`` for
a transport-incomplete frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class TwoStageReason(str, Enum):
    RESULT_PUBLISHED = "RESULT_PUBLISHED"
    TRANSPORT_INCOMPLETE = "TRANSPORT_INCOMPLETE"
    MEASURED_PRE_QUEUE_REJECTION = "MEASURED_PRE_QUEUE_REJECTION"
    SUPERSEDED_PENDING = "SUPERSEDED_PENDING"
    QUEUE_WAIT_BUDGET_EXCEEDED = "QUEUE_WAIT_BUDGET_EXCEEDED"
    PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL = (
        "PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL"
    )
    PROCESSING_HORIZON_EXPIRED_BEFORE_COMPUTE = (
        "PROCESSING_HORIZON_EXPIRED_BEFORE_COMPUTE"
    )
    PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED = (
        "PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED"
    )
    PROCESSING_HORIZON_EXPIRED_AFTER_COMPUTE = (
        "PROCESSING_HORIZON_EXPIRED_AFTER_COMPUTE"
    )
    SUPERSEDED_PUBLICATION_PENDING = "SUPERSEDED_PUBLICATION_PENDING"
    PROCESSING_HORIZON_EXPIRED_BEFORE_PUBLICATION = (
        "PROCESSING_HORIZON_EXPIRED_BEFORE_PUBLICATION"
    )
    PROCESSING_HORIZON_EXPIRED_AFTER_PUBLICATION = (
        "PROCESSING_HORIZON_EXPIRED_AFTER_PUBLICATION"
    )


@dataclass(frozen=True)
class TwoStageFrame:
    frame_id: int
    sequence_id: int
    capture_ns: int
    arrival_ns: int | None
    compute_ns: int
    publication_ns: int
    post_publication_install_ns: int
    feature_bytes: int
    pre_scheduler_reason: TwoStageReason | None = None
    service_observed: bool = True
    install_delay_observed: bool = True

    def __post_init__(self) -> None:
        _require(self.frame_id >= 0, "frame_id must be non-negative")
        _require(self.sequence_id >= 0, "sequence_id must be non-negative")
        _require(self.capture_ns >= 0, "capture_ns must be non-negative")
        if self.arrival_ns is not None:
            _require(self.arrival_ns >= self.capture_ns, "arrival precedes capture")
        if self.pre_scheduler_reason is not None:
            _require(self.arrival_ns is None, "pre-terminal frame has an arrival")
            _require(
                self.pre_scheduler_reason
                in {
                    TwoStageReason.TRANSPORT_INCOMPLETE,
                    TwoStageReason.MEASURED_PRE_QUEUE_REJECTION,
                },
                "invalid pre-scheduler terminal reason",
            )
        _require(self.compute_ns > 0, "compute_ns must be positive")
        _require(self.publication_ns > 0, "publication_ns must be positive")
        _require(
            self.post_publication_install_ns >= 0,
            "post-publication install delay must be non-negative",
        )
        _require(self.feature_bytes >= 0, "feature_bytes must be non-negative")


@dataclass(frozen=True)
class TwoStageConfig:
    queue_wait_budget_ns: int | None
    processing_horizon_ns: int = 500_000_000
    service_target_ns: int = 100_000_000
    predicted_compute_ns: int | None = None
    predicted_publication_ns: int = 0
    predicted_post_publication_install_ns: int = 0

    def __post_init__(self) -> None:
        if self.queue_wait_budget_ns is not None:
            _require(self.queue_wait_budget_ns >= 0, "queue budget is invalid")
        _require(self.processing_horizon_ns > 0, "processing horizon is invalid")
        _require(self.service_target_ns > 0, "service target is invalid")
        if self.predicted_compute_ns is not None:
            _require(self.predicted_compute_ns > 0, "compute prediction is invalid")
            _require(
                self.predicted_publication_ns >= 0,
                "publication prediction is invalid",
            )
            _require(
                self.predicted_post_publication_install_ns >= 0,
                "install-delay prediction is invalid",
            )
        else:
            _require(
                self.predicted_publication_ns == 0
                and self.predicted_post_publication_install_ns == 0,
                "downstream predictions require a compute prediction",
            )


@dataclass(frozen=True)
class TwoStageOutcome:
    frame: TwoStageFrame
    reason: TwoStageReason
    terminal_ns: int
    compute_start_ns: int | None = None
    compute_finish_ns: int | None = None
    publication_start_ns: int | None = None
    publication_finish_ns: int | None = None
    install_ns: int | None = None
    replaced_by_sequence_id: int | None = None

    @property
    def queue_wait_ns(self) -> int | None:
        if self.compute_start_ns is None or self.frame.arrival_ns is None:
            return None
        return self.compute_start_ns - self.frame.arrival_ns

    @property
    def install_aoi_ns(self) -> int | None:
        if self.install_ns is None:
            return None
        return self.install_ns - self.frame.capture_ns

    @property
    def compute_spent_ns(self) -> int:
        if self.compute_start_ns is None:
            return 0
        finish = self.compute_finish_ns
        if finish is None:
            finish = self.terminal_ns
        return max(0, int(finish) - int(self.compute_start_ns))

    @property
    def publication_spent_ns(self) -> int:
        if self.publication_start_ns is None:
            return 0
        finish = self.publication_finish_ns
        if finish is None:
            finish = self.terminal_ns
        return max(0, int(finish) - int(self.publication_start_ns))


@dataclass(frozen=True)
class TwoStageResult:
    config: TwoStageConfig
    outcomes: tuple[TwoStageOutcome, ...]
    observation_start_ns: int
    observation_end_ns: int

    def summary(self) -> dict[str, Any]:
        by_reason = {
            reason.value: sum(item.reason is reason for item in self.outcomes)
            for reason in TwoStageReason
        }
        published = [
            item
            for item in self.outcomes
            if item.reason is TwoStageReason.RESULT_PUBLISHED
        ]
        installed = [item for item in published if item.install_ns is not None]
        installed.sort(key=lambda item: (int(item.install_ns), item.frame.sequence_id))

        useful: list[TwoStageOutcome] = []
        newest_capture_ns = -1
        for item in installed:
            if item.frame.capture_ns > newest_capture_ns:
                useful.append(item)
                newest_capture_ns = item.frame.capture_ns

        install_aoi_ms = [
            float(item.install_aoi_ns) / 1_000_000.0
            for item in installed
            if item.install_aoi_ns is not None
        ]
        useful_aoi_ms = [
            float(item.install_aoi_ns) / 1_000_000.0
            for item in useful
            if item.install_aoi_ns is not None
        ]
        queue_wait_ms = [
            float(item.queue_wait_ns) / 1_000_000.0
            for item in self.outcomes
            if item.queue_wait_ns is not None
        ]
        total = len(self.outcomes)
        scheduler_inputs = [
            item for item in self.outcomes if item.frame.arrival_ns is not None
        ]
        useful_sequences = {item.frame.sequence_id for item in useful}
        nonuseful = [
            item
            for item in self.outcomes
            if item.frame.sequence_id not in useful_sequences
        ]
        aoi = _time_weighted_aoi(
            useful,
            start_ns=self.observation_start_ns,
            end_ns=self.observation_end_ns,
            threshold_ns=self.config.service_target_ns,
        )
        return {
            "input_frames": total,
            "edge_scheduler_input_frames": len(scheduler_inputs),
            "queue_wait_budget_ms": (
                None
                if self.config.queue_wait_budget_ns is None
                else self.config.queue_wait_budget_ns / 1_000_000.0
            ),
            "processing_horizon_ms": self.config.processing_horizon_ns / 1e6,
            "service_target_ms": self.config.service_target_ns / 1e6,
            "predicted_compute_ms": (
                None
                if self.config.predicted_compute_ns is None
                else self.config.predicted_compute_ns / 1e6
            ),
            "predicted_publication_ms": self.config.predicted_publication_ns / 1e6,
            "predicted_post_publication_install_ms": (
                self.config.predicted_post_publication_install_ns / 1e6
            ),
            "reason_counts": by_reason,
            "edge_results_published": len(published),
            "ack_installed_frames": len(installed),
            "useful_newer_map_installations": len(useful),
            "installed_within_100ms": sum(value <= 100.0 for value in install_aoi_ms),
            "installed_within_500ms": sum(value <= 500.0 for value in install_aoi_ms),
            "install_aoi_ms_median": _percentile(install_aoi_ms, 0.5),
            "install_aoi_ms_p95": _percentile(install_aoi_ms, 0.95),
            "useful_install_aoi_ms_median": _percentile(useful_aoi_ms, 0.5),
            "queue_wait_ms_median": _percentile(queue_wait_ms, 0.5),
            "queue_wait_ms_p95": _percentile(queue_wait_ms, 0.95),
            "feature_bytes_charged": sum(item.frame.feature_bytes for item in self.outcomes),
            "feature_bytes_without_useful_install": sum(
                item.frame.feature_bytes for item in nonuseful
            ),
            "compute_ms_charged": sum(item.compute_spent_ns for item in self.outcomes)
            / 1e6,
            "publication_ms_charged": sum(
                item.publication_spent_ns for item in self.outcomes
            )
            / 1e6,
            "service_observed_fraction_among_scheduler_inputs": (
                None
                if not scheduler_inputs
                else sum(item.frame.service_observed for item in scheduler_inputs)
                / len(scheduler_inputs)
            ),
            "install_delay_observed_fraction_among_scheduler_inputs": (
                None
                if not scheduler_inputs
                else sum(item.frame.install_delay_observed for item in scheduler_inputs)
                / len(scheduler_inputs)
            ),
            **aoi,
        }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _time_weighted_aoi(
    installed: list[TwoStageOutcome],
    *,
    start_ns: int,
    end_ns: int,
    threshold_ns: int,
) -> dict[str, float | None]:
    relevant = [
        item
        for item in installed
        if item.install_ns is not None and int(item.install_ns) < end_ns
    ]
    if not relevant:
        return {
            "time_weighted_map_aoi_ms": None,
            "map_time_above_service_target_fraction": None,
            "map_aoi_observation_duration_s": 0.0,
        }
    evaluation_start = max(start_ns, int(relevant[0].install_ns))
    if evaluation_start >= end_ns:
        return {
            "time_weighted_map_aoi_ms": None,
            "map_time_above_service_target_fraction": None,
            "map_aoi_observation_duration_s": 0.0,
        }
    area_ns2 = 0.0
    above_ns = 0
    for index, item in enumerate(relevant):
        interval_start = max(evaluation_start, int(item.install_ns))
        interval_end = (
            min(end_ns, int(relevant[index + 1].install_ns))
            if index + 1 < len(relevant)
            else end_ns
        )
        if interval_end <= interval_start:
            continue
        start_age = interval_start - item.frame.capture_ns
        duration = interval_end - interval_start
        area_ns2 += float(start_age) * duration + 0.5 * float(duration) ** 2
        crossing = item.frame.capture_ns + int(threshold_ns)
        above_ns += max(0, interval_end - max(interval_start, crossing))
    observation_ns = end_ns - evaluation_start
    return {
        "time_weighted_map_aoi_ms": area_ns2 / observation_ns / 1_000_000.0,
        "map_time_above_service_target_fraction": above_ns / observation_ns,
        "map_aoi_observation_duration_s": observation_ns / 1e9,
    }


def simulate_two_stage(
    frames: Iterable[TwoStageFrame], *, config: TwoStageConfig
) -> TwoStageResult:
    offered = list(frames)
    _require(bool(offered), "at least one frame is required")
    identities = {(item.frame_id, item.sequence_id) for item in offered}
    _require(len(identities) == len(offered), "duplicate frame identity")
    ordered = sorted(
        (item for item in offered if item.arrival_ns is not None),
        key=lambda item: (int(item.arrival_ns), item.sequence_id),
    )
    outcomes: dict[int, TwoStageOutcome] = {}
    for item in offered:
        if item.arrival_ns is None:
            outcomes[item.sequence_id] = TwoStageOutcome(
                frame=item,
                reason=(
                    item.pre_scheduler_reason
                    or TwoStageReason.TRANSPORT_INCOMPLETE
                ),
                terminal_ns=item.capture_ns + config.processing_horizon_ns,
            )

    pending_compute: TwoStageFrame | None = None
    active_compute: tuple[TwoStageFrame, int, int] | None = None
    pending_publication: tuple[TwoStageFrame, int, int] | None = None
    active_publication: tuple[TwoStageFrame, int, int, int, int] | None = None
    index = 0

    def terminal(
        item: TwoStageFrame,
        reason: TwoStageReason,
        at_ns: int,
        *,
        compute_start_ns: int | None = None,
        compute_finish_ns: int | None = None,
        publication_start_ns: int | None = None,
        publication_finish_ns: int | None = None,
        install_ns: int | None = None,
        replaced_by_sequence_id: int | None = None,
    ) -> None:
        _require(item.sequence_id not in outcomes, "frame received two terminals")
        outcomes[item.sequence_id] = TwoStageOutcome(
            frame=item,
            reason=reason,
            terminal_ns=int(at_ns),
            compute_start_ns=compute_start_ns,
            compute_finish_ns=compute_finish_ns,
            publication_start_ns=publication_start_ns,
            publication_finish_ns=publication_finish_ns,
            install_ns=install_ns,
            replaced_by_sequence_id=replaced_by_sequence_id,
        )

    def start_compute(at_ns: int) -> None:
        nonlocal pending_compute, active_compute
        while active_compute is None and pending_compute is not None:
            item = pending_compute
            pending_compute = None
            wait_ns = at_ns - int(item.arrival_ns)
            if (
                config.queue_wait_budget_ns is not None
                and wait_ns > config.queue_wait_budget_ns
            ):
                terminal(item, TwoStageReason.QUEUE_WAIT_BUDGET_EXCEEDED, at_ns)
                continue
            if at_ns - item.capture_ns > config.processing_horizon_ns:
                terminal(
                    item,
                    TwoStageReason.PROCESSING_HORIZON_EXPIRED_BEFORE_COMPUTE,
                    at_ns,
                )
                continue
            if config.predicted_compute_ns is not None:
                predicted_compute_finish = at_ns + config.predicted_compute_ns
                publication_available = (
                    at_ns
                    if active_publication is None
                    else active_publication[4]
                )
                predicted_publication_start = max(
                    predicted_compute_finish, publication_available
                )
                predicted_install = (
                    predicted_publication_start
                    + config.predicted_publication_ns
                    + config.predicted_post_publication_install_ns
                )
                if predicted_install - item.capture_ns > config.processing_horizon_ns:
                    terminal(
                        item,
                        TwoStageReason.PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED,
                        at_ns,
                    )
                    continue
            active_compute = (item, at_ns, at_ns + item.compute_ns)

    def enqueue_publication(
        item: TwoStageFrame, compute_start: int, compute_finish: int, at_ns: int
    ) -> None:
        nonlocal pending_publication
        if at_ns - item.capture_ns > config.processing_horizon_ns:
            terminal(
                item,
                TwoStageReason.PROCESSING_HORIZON_EXPIRED_AFTER_COMPUTE,
                at_ns,
                compute_start_ns=compute_start,
                compute_finish_ns=compute_finish,
            )
            return
        if pending_publication is not None:
            old, old_compute_start, old_compute_finish = pending_publication
            terminal(
                old,
                TwoStageReason.SUPERSEDED_PUBLICATION_PENDING,
                at_ns,
                compute_start_ns=old_compute_start,
                compute_finish_ns=old_compute_finish,
                replaced_by_sequence_id=item.sequence_id,
            )
        pending_publication = (item, compute_start, compute_finish)

    def start_publication(at_ns: int) -> None:
        nonlocal pending_publication, active_publication
        while active_publication is None and pending_publication is not None:
            item, compute_start, compute_finish = pending_publication
            pending_publication = None
            if at_ns - item.capture_ns > config.processing_horizon_ns:
                terminal(
                    item,
                    TwoStageReason.PROCESSING_HORIZON_EXPIRED_BEFORE_PUBLICATION,
                    at_ns,
                    compute_start_ns=compute_start,
                    compute_finish_ns=compute_finish,
                )
                continue
            active_publication = (
                item,
                compute_start,
                compute_finish,
                at_ns,
                at_ns + item.publication_ns,
            )

    while (
        index < len(ordered)
        or pending_compute is not None
        or active_compute is not None
        or pending_publication is not None
        or active_publication is not None
    ):
        event_times: list[int] = []
        if index < len(ordered):
            event_times.append(int(ordered[index].arrival_ns))
        if active_compute is not None:
            event_times.append(active_compute[2])
        if active_publication is not None:
            event_times.append(active_publication[4])
        _require(bool(event_times), "event loop stalled")
        now_ns = min(event_times)

        # Complete active work before admitting arrivals with the same stamp.
        if active_compute is not None and active_compute[2] == now_ns:
            item, compute_start, compute_finish = active_compute
            active_compute = None
            enqueue_publication(item, compute_start, compute_finish, now_ns)
        if active_publication is not None and active_publication[4] == now_ns:
            item, compute_start, compute_finish, pub_start, pub_finish = (
                active_publication
            )
            active_publication = None
            if now_ns - item.capture_ns > config.processing_horizon_ns:
                terminal(
                    item,
                    TwoStageReason.PROCESSING_HORIZON_EXPIRED_AFTER_PUBLICATION,
                    now_ns,
                    compute_start_ns=compute_start,
                    compute_finish_ns=compute_finish,
                    publication_start_ns=pub_start,
                    publication_finish_ns=pub_finish,
                )
            else:
                terminal(
                    item,
                    TwoStageReason.RESULT_PUBLISHED,
                    now_ns,
                    compute_start_ns=compute_start,
                    compute_finish_ns=compute_finish,
                    publication_start_ns=pub_start,
                    publication_finish_ns=pub_finish,
                    install_ns=now_ns + item.post_publication_install_ns,
                )

        while index < len(ordered) and int(ordered[index].arrival_ns) == now_ns:
            item = ordered[index]
            index += 1
            if now_ns - item.capture_ns > config.processing_horizon_ns:
                terminal(
                    item,
                    TwoStageReason.PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL,
                    now_ns,
                )
                continue
            if pending_compute is not None:
                terminal(
                    pending_compute,
                    TwoStageReason.SUPERSEDED_PENDING,
                    now_ns,
                    replaced_by_sequence_id=item.sequence_id,
                )
            pending_compute = item

        start_publication(now_ns)
        start_compute(now_ns)

    _require(len(outcomes) == len(offered), "terminal accounting is incomplete")
    ordered_outcomes = tuple(outcomes[item.sequence_id] for item in offered)
    observation_start = min(item.capture_ns for item in offered)
    observation_end = max(item.capture_ns for item in offered) + config.processing_horizon_ns
    return TwoStageResult(
        config=config,
        outcomes=ordered_outcomes,
        observation_start_ns=observation_start,
        observation_end_ns=observation_end,
    )

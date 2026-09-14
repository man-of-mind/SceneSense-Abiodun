"""Deterministic two-stage queue models for post-campaign analysis.

The two policies deliberately share every timing input.  They differ only in
what happens to pending work:

* ``FIFO_NO_DISCARD`` retains every admitted frame and drains it in order.
* ``LATEST_ONLY_NO_EXPIRY`` keeps one newest pending item at each stage.

Active compute/publication is non-preemptive in both policies.  There is no
queue-wait expiry and no predicted completion gate in either model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageFrame,
    TwoStageReason,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class QueuePolicy(str, Enum):
    FIFO_NO_DISCARD = "FIFO_NO_DISCARD"
    LATEST_ONLY_NO_EXPIRY = "LATEST_ONLY_NO_EXPIRY"


class QueueReason(str, Enum):
    RESULT_PUBLISHED = "RESULT_PUBLISHED"
    TRANSPORT_INCOMPLETE = "TRANSPORT_INCOMPLETE"
    MEASURED_PRE_QUEUE_REJECTION = "MEASURED_PRE_QUEUE_REJECTION"
    SUPERSEDED_PENDING_COMPUTE = "SUPERSEDED_PENDING_COMPUTE"
    SUPERSEDED_PENDING_PUBLICATION = "SUPERSEDED_PENDING_PUBLICATION"


@dataclass(frozen=True)
class QueueOutcome:
    frame: TwoStageFrame
    reason: QueueReason
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
        end = self.compute_finish_ns or self.terminal_ns
        return max(0, int(end) - int(self.compute_start_ns))

    @property
    def publication_spent_ns(self) -> int:
        if self.publication_start_ns is None:
            return 0
        end = self.publication_finish_ns or self.terminal_ns
        return max(0, int(end) - int(self.publication_start_ns))


@dataclass(frozen=True)
class QueueSimulation:
    policy: QueuePolicy
    outcomes: tuple[QueueOutcome, ...]
    observation_start_ns: int
    observation_end_ns: int
    compute_queue_high_water: int
    publication_queue_high_water: int


PublicationTicket = tuple[TwoStageFrame, int, int]


def simulate_queue_policy(
    frames: Iterable[TwoStageFrame],
    *,
    policy: QueuePolicy,
    observation_tail_ns: int = 500_000_000,
) -> QueueSimulation:
    """Run one no-expiry, non-preemptive two-stage policy to full drain."""

    offered = list(frames)
    _require(bool(offered), "at least one frame is required")
    _require(observation_tail_ns >= 0, "observation tail must be non-negative")
    identities = {(item.frame_id, item.sequence_id) for item in offered}
    _require(len(identities) == len(offered), "duplicate frame identity")

    ordered = sorted(
        (item for item in offered if item.arrival_ns is not None),
        key=lambda item: (int(item.arrival_ns), item.sequence_id),
    )
    outcomes: dict[int, QueueOutcome] = {}
    for item in offered:
        if item.arrival_ns is not None:
            continue
        reason = (
            QueueReason.MEASURED_PRE_QUEUE_REJECTION
            if item.pre_scheduler_reason
            is TwoStageReason.MEASURED_PRE_QUEUE_REJECTION
            else QueueReason.TRANSPORT_INCOMPLETE
        )
        outcomes[item.sequence_id] = QueueOutcome(
            frame=item,
            reason=reason,
            terminal_ns=item.capture_ns + observation_tail_ns,
        )

    compute_fifo: deque[TwoStageFrame] = deque()
    publication_fifo: deque[PublicationTicket] = deque()
    compute_latest: TwoStageFrame | None = None
    publication_latest: PublicationTicket | None = None
    active_compute: tuple[TwoStageFrame, int, int] | None = None
    active_publication: tuple[TwoStageFrame, int, int, int, int] | None = None
    compute_high_water = 0
    publication_high_water = 0
    index = 0

    def terminal(
        item: TwoStageFrame,
        reason: QueueReason,
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
        outcomes[item.sequence_id] = QueueOutcome(
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

    def enqueue_compute(item: TwoStageFrame, at_ns: int) -> None:
        nonlocal compute_latest, compute_high_water
        if policy is QueuePolicy.FIFO_NO_DISCARD:
            compute_fifo.append(item)
            compute_high_water = max(compute_high_water, len(compute_fifo))
            return
        if compute_latest is not None:
            terminal(
                compute_latest,
                QueueReason.SUPERSEDED_PENDING_COMPUTE,
                at_ns,
                replaced_by_sequence_id=item.sequence_id,
            )
        compute_latest = item
        compute_high_water = max(compute_high_water, 1)

    def pop_compute() -> TwoStageFrame | None:
        nonlocal compute_latest
        if policy is QueuePolicy.FIFO_NO_DISCARD:
            return compute_fifo.popleft() if compute_fifo else None
        item = compute_latest
        compute_latest = None
        return item

    def enqueue_publication(ticket: PublicationTicket, at_ns: int) -> None:
        nonlocal publication_latest, publication_high_water
        if policy is QueuePolicy.FIFO_NO_DISCARD:
            publication_fifo.append(ticket)
            publication_high_water = max(
                publication_high_water, len(publication_fifo)
            )
            return
        if publication_latest is not None:
            old, old_start, old_finish = publication_latest
            terminal(
                old,
                QueueReason.SUPERSEDED_PENDING_PUBLICATION,
                at_ns,
                compute_start_ns=old_start,
                compute_finish_ns=old_finish,
                replaced_by_sequence_id=ticket[0].sequence_id,
            )
        publication_latest = ticket
        publication_high_water = max(publication_high_water, 1)

    def pop_publication() -> PublicationTicket | None:
        nonlocal publication_latest
        if policy is QueuePolicy.FIFO_NO_DISCARD:
            return publication_fifo.popleft() if publication_fifo else None
        ticket = publication_latest
        publication_latest = None
        return ticket

    def start_compute(at_ns: int) -> None:
        nonlocal active_compute
        if active_compute is not None:
            return
        item = pop_compute()
        if item is not None:
            active_compute = (item, at_ns, at_ns + item.compute_ns)

    def start_publication(at_ns: int) -> None:
        nonlocal active_publication
        if active_publication is not None:
            return
        ticket = pop_publication()
        if ticket is None:
            return
        item, compute_start, compute_finish = ticket
        active_publication = (
            item,
            compute_start,
            compute_finish,
            at_ns,
            at_ns + item.publication_ns,
        )

    def compute_pending() -> bool:
        return bool(compute_fifo) if policy is QueuePolicy.FIFO_NO_DISCARD else compute_latest is not None

    def publication_pending() -> bool:
        return bool(publication_fifo) if policy is QueuePolicy.FIFO_NO_DISCARD else publication_latest is not None

    while (
        index < len(ordered)
        or compute_pending()
        or active_compute is not None
        or publication_pending()
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

        # Finish active non-preemptive work before same-timestamp arrivals.
        if active_compute is not None and active_compute[2] == now_ns:
            item, compute_start, compute_finish = active_compute
            active_compute = None
            enqueue_publication((item, compute_start, compute_finish), now_ns)
        if active_publication is not None and active_publication[4] == now_ns:
            item, compute_start, compute_finish, pub_start, pub_finish = active_publication
            active_publication = None
            terminal(
                item,
                QueueReason.RESULT_PUBLISHED,
                now_ns,
                compute_start_ns=compute_start,
                compute_finish_ns=compute_finish,
                publication_start_ns=pub_start,
                publication_finish_ns=pub_finish,
                install_ns=now_ns + item.post_publication_install_ns,
            )

        while index < len(ordered) and int(ordered[index].arrival_ns) == now_ns:
            enqueue_compute(ordered[index], now_ns)
            index += 1

        start_publication(now_ns)
        start_compute(now_ns)

    _require(len(outcomes) == len(offered), "terminal accounting is incomplete")
    observation_start_ns = min(item.capture_ns for item in offered)
    observation_end_ns = (
        max(item.capture_ns for item in offered) + observation_tail_ns
    )
    return QueueSimulation(
        policy=policy,
        outcomes=tuple(outcomes[item.sequence_id] for item in offered),
        observation_start_ns=observation_start_ns,
        observation_end_ns=observation_end_ns,
        compute_queue_high_water=compute_high_water,
        publication_queue_high_water=publication_high_water,
    )

"""Bounded two-stage execution candidate for the SplitFusion edge.

The first worker is the sole owner of decode/tail/GPU work.  The second worker
owns CPU-only publication.  Each boundary has one latest-only pending slot;
active callbacks are never interrupted.

This module contains no CUDA, networking, or model construction.  The live
runtime can inject those operations only after the concurrency and parity
contract has been qualified.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .scheduler import (
    Admission,
    FrameTicket,
    OutcomeClass,
    Stage,
    TerminalFeedback,
    TerminalReason,
)


MS = 1_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class CandidatePolicy(str, Enum):
    LATEST_ONLY_NO_EXPIRY = "LATEST_ONLY_NO_EXPIRY"
    LATEST_ONLY_25_MS = "LATEST_ONLY_25_MS"

    @property
    def queue_wait_budget_ns(self) -> int | None:
        if self is CandidatePolicy.LATEST_ONLY_25_MS:
            return 25 * MS
        return None


@dataclass(frozen=True)
class PipelineConfig:
    policy: CandidatePolicy
    processing_horizon_ns: int = 500 * MS

    def __post_init__(self) -> None:
        _require(
            isinstance(self.policy, CandidatePolicy),
            "policy must be a registered candidate",
        )
        _require(self.processing_horizon_ns > 0, "processing horizon is invalid")

    @property
    def queue_wait_budget_ns(self) -> int | None:
        return self.policy.queue_wait_budget_ns


@dataclass(frozen=True)
class PipelineSnapshot:
    policy: str
    accepting: bool
    compute_pending_depth: int
    publication_pending_depth: int
    compute_active: bool
    publication_active: bool
    offered_frames: int
    compute_started: int
    compute_completed: int
    publication_started: int
    publication_completed: int
    terminal_frames: int
    compute_pending_high_water: int
    publication_pending_high_water: int
    maximum_active_stage_workers: int
    stage_overlap_observed: bool
    compute_owner_thread_id: int | None
    publication_owner_thread_id: int | None
    fatal_error: str | None


@dataclass(frozen=True)
class _WorkItem:
    ticket: FrameTicket
    payload: Any


@dataclass(frozen=True)
class _ComputedItem:
    ticket: FrameTicket
    value: Any
    compute_started_ns: int
    compute_finished_ns: int

    @property
    def compute_spent_ns(self) -> int:
        return self.compute_finished_ns - self.compute_started_ns

    @property
    def queue_wait_ns(self) -> int:
        return self.compute_started_ns - self.ticket.edge_arrival_timestamp_ns


class PipelineWorkerError(RuntimeError):
    """Raised by ``close_and_join`` after a worker callback fails."""


class BoundedTwoStagePipeline:
    """One compute worker plus one publication worker and latest-only slots.

    A pipeline instance is bound to the first ticket's run, cell, and stream.
    This makes it a per-UE/per-cell execution primitive rather than a hidden
    cross-UE scheduler.  Multi-UE fairness remains an explicit upper layer.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig,
        compute: Callable[[FrameTicket, Any], Any],
        publish: Callable[[FrameTicket, Any], Any],
        feedback_sink: Callable[[TerminalFeedback], None] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        _require(callable(compute), "compute callback is required")
        _require(callable(publish), "publish callback is required")
        _require(callable(clock_ns), "clock callback is required")
        self._config = config
        self._compute = compute
        self._publish = publish
        self._feedback_sink = feedback_sink
        self._clock_ns = clock_ns
        self._condition = threading.Condition(threading.RLock())
        self._compute_pending: _WorkItem | None = None
        self._publication_pending: _ComputedItem | None = None
        self._compute_active: _WorkItem | None = None
        self._publication_active: _ComputedItem | None = None
        self._binding: tuple[str, str, str] | None = None
        self._latest_sequence_id = -1
        self._latest_ticket: FrameTicket | None = None
        self._terminal: dict[tuple[str, str, str, int, int], TerminalFeedback] = {}
        self._started = False
        self._accepting = False
        self._closing = False
        self._compute_done = False
        self._fatal_error: BaseException | None = None
        self._compute_thread: threading.Thread | None = None
        self._publication_thread: threading.Thread | None = None
        self._compute_owner_thread_id: int | None = None
        self._publication_owner_thread_id: int | None = None
        self._offered_frames = 0
        self._compute_started = 0
        self._compute_completed = 0
        self._publication_started = 0
        self._publication_completed = 0
        self._compute_pending_high_water = 0
        self._publication_pending_high_water = 0
        self._maximum_active_stage_workers = 0
        self._stage_overlap_observed = False

    def start(self) -> None:
        with self._condition:
            _require(not self._started, "pipeline already started")
            self._started = True
            self._accepting = True
            self._compute_thread = threading.Thread(
                target=self._compute_loop,
                name="splitfusion-edge-compute-owner",
                daemon=False,
            )
            self._publication_thread = threading.Thread(
                target=self._publication_loop,
                name="splitfusion-edge-publication-owner",
                daemon=False,
            )
            self._compute_thread.start()
            self._publication_thread.start()

    def _now(self) -> int:
        value = int(self._clock_ns())
        _require(value >= 0, "clock returned a negative timestamp")
        return value

    def _feedback_locked(
        self,
        ticket: FrameTicket,
        *,
        reason: TerminalReason,
        outcome_class: OutcomeClass,
        stage: Stage,
        now_ns: int,
        queue_wait_ns: int,
        compute_spent_ns: int = 0,
        publication_spent_ns: int = 0,
        replacement: FrameTicket | None = None,
    ) -> TerminalFeedback:
        _require(ticket.identity not in self._terminal, "frame already terminal")
        feedback = TerminalFeedback(
            ticket=ticket,
            reason=reason,
            outcome_class=outcome_class,
            stage=stage,
            observed_timestamp_ns=now_ns,
            queue_wait_ns=max(0, int(queue_wait_ns)),
            bytes_already_sent=ticket.feature_bytes,
            compute_spent_ns=max(0, int(compute_spent_ns)),
            publication_spent_ns=max(0, int(publication_spent_ns)),
            replacing_frame_id=(None if replacement is None else replacement.frame_id),
            replacing_sequence_id=(
                None if replacement is None else replacement.sequence_id
            ),
        )
        self._terminal[ticket.identity] = feedback
        return feedback

    def _deliver(self, feedback: list[TerminalFeedback]) -> bool:
        if self._feedback_sink is None:
            return True
        try:
            for item in feedback:
                self._feedback_sink(item)
        except Exception as exc:
            now_ns = max(
                [time.time_ns()]
                + [item.observed_timestamp_ns for item in feedback]
            )
            with self._condition:
                self._fail_locked(
                    RuntimeError(f"terminal feedback sink failed: {exc}"), now_ns
                )
            return False
        return True

    def offer(
        self,
        ticket: FrameTicket,
        payload: Any,
        *,
        now_ns: int | None = None,
    ) -> Admission:
        observed_ns = self._now() if now_ns is None else int(now_ns)
        feedback: list[TerminalFeedback] = []
        with self._condition:
            _require(self._started, "pipeline is not started")
            _require(self._accepting, "pipeline is not accepting work")
            _require(
                observed_ns >= ticket.edge_arrival_timestamp_ns,
                "offer time precedes edge arrival",
            )
            binding = (ticket.run_id, ticket.cell_id, ticket.stream_id)
            if self._binding is None:
                self._binding = binding
            _require(binding == self._binding, "ticket crosses pipeline binding")
            self._offered_frames += 1
            if ticket.sequence_id <= self._latest_sequence_id:
                latest = self._latest_ticket
                assert latest is not None
                feedback.append(
                    self._feedback_locked(
                        ticket,
                        reason=TerminalReason.OUT_OF_ORDER_ARRIVAL,
                        outcome_class=OutcomeClass.INTENTIONAL_FRESHNESS_DROP,
                        stage=Stage.PENDING,
                        now_ns=observed_ns,
                        queue_wait_ns=observed_ns
                        - ticket.edge_arrival_timestamp_ns,
                        replacement=latest,
                    )
                )
                admitted = False
            else:
                self._latest_sequence_id = ticket.sequence_id
                self._latest_ticket = ticket
                if self._compute_pending is not None:
                    displaced = self._compute_pending.ticket
                    feedback.append(
                        self._feedback_locked(
                            displaced,
                            reason=TerminalReason.SUPERSEDED_PENDING,
                            outcome_class=OutcomeClass.INTENTIONAL_FRESHNESS_DROP,
                            stage=Stage.PENDING,
                            now_ns=observed_ns,
                            queue_wait_ns=observed_ns
                            - displaced.edge_arrival_timestamp_ns,
                            replacement=ticket,
                        )
                    )
                self._compute_pending = _WorkItem(ticket, payload)
                self._compute_pending_high_water = 1
                self._condition.notify_all()
                admitted = True
        if not self._deliver(feedback):
            raise PipelineWorkerError("terminal feedback sink failed")
        return Admission(admitted, tuple(feedback))

    def _mark_stage_active_locked(self) -> None:
        active = int(self._compute_active is not None) + int(
            self._publication_active is not None
        )
        self._maximum_active_stage_workers = max(
            self._maximum_active_stage_workers, active
        )
        if active == 2:
            self._stage_overlap_observed = True

    def _abort_pending_locked(self, now_ns: int) -> list[TerminalFeedback]:
        feedback: list[TerminalFeedback] = []
        if self._compute_pending is not None:
            ticket = self._compute_pending.ticket
            self._compute_pending = None
            feedback.append(
                self._feedback_locked(
                    ticket,
                    reason=TerminalReason.PIPELINE_ABORTED,
                    outcome_class=OutcomeClass.STRUCTURAL_FAILURE,
                    stage=Stage.PENDING,
                    now_ns=max(now_ns, ticket.edge_arrival_timestamp_ns),
                    queue_wait_ns=max(0, now_ns - ticket.edge_arrival_timestamp_ns),
                )
            )
        if self._publication_pending is not None:
            item = self._publication_pending
            self._publication_pending = None
            feedback.append(
                self._feedback_locked(
                    item.ticket,
                    reason=TerminalReason.PIPELINE_ABORTED,
                    outcome_class=OutcomeClass.STRUCTURAL_FAILURE,
                    stage=Stage.PENDING_PUBLICATION,
                    now_ns=max(now_ns, item.ticket.edge_arrival_timestamp_ns),
                    queue_wait_ns=item.queue_wait_ns,
                    compute_spent_ns=item.compute_spent_ns,
                )
            )
        return feedback

    def _fail_locked(self, error: BaseException, now_ns: int) -> list[TerminalFeedback]:
        if self._fatal_error is None:
            self._fatal_error = error
        self._accepting = False
        self._closing = True
        feedback = self._abort_pending_locked(now_ns)
        self._condition.notify_all()
        return feedback

    def _compute_loop(self) -> None:
        feedback: list[TerminalFeedback] = []
        try:
            with self._condition:
                owner = threading.get_ident()
                self._compute_owner_thread_id = owner
            while True:
                feedback = []
                with self._condition:
                    while (
                        self._compute_pending is None
                        and not self._closing
                        and self._fatal_error is None
                    ):
                        self._condition.wait()
                    if self._fatal_error is not None:
                        break
                    if self._compute_pending is None and self._closing:
                        break
                    item = self._compute_pending
                    assert item is not None
                    self._compute_pending = None
                    self._compute_active = item
                started_ns = self._now()
                queue_wait_ns = started_ns - item.ticket.edge_arrival_timestamp_ns
                budget = self._config.queue_wait_budget_ns
                if budget is not None and queue_wait_ns > budget:
                    with self._condition:
                        feedback.append(
                            self._feedback_locked(
                                item.ticket,
                                reason=TerminalReason.QUEUE_WAIT_BUDGET_EXCEEDED,
                                outcome_class=OutcomeClass.EXPIRED_WORK,
                                stage=Stage.PENDING,
                                now_ns=started_ns,
                                queue_wait_ns=queue_wait_ns,
                            )
                        )
                        self._compute_active = None
                    self._deliver(feedback)
                    continue
                if (
                    started_ns - item.ticket.capture_timestamp_ns
                    > self._config.processing_horizon_ns
                ):
                    with self._condition:
                        feedback.append(
                            self._feedback_locked(
                                item.ticket,
                                reason=TerminalReason.PROCESSING_HORIZON_EXPIRED,
                                outcome_class=OutcomeClass.EXPIRED_WORK,
                                stage=Stage.BEFORE_DECODE,
                                now_ns=started_ns,
                                queue_wait_ns=queue_wait_ns,
                            )
                        )
                        self._compute_active = None
                    self._deliver(feedback)
                    continue
                with self._condition:
                    self._compute_started += 1
                    self._mark_stage_active_locked()
                try:
                    value = self._compute(item.ticket, item.payload)
                except BaseException as exc:
                    failed_ns = self._now()
                    with self._condition:
                        feedback.append(
                            self._feedback_locked(
                                item.ticket,
                                reason=TerminalReason.PROCESSING_FAILED,
                                outcome_class=OutcomeClass.STRUCTURAL_FAILURE,
                                stage=Stage.BEFORE_PUBLICATION,
                                now_ns=failed_ns,
                                queue_wait_ns=queue_wait_ns,
                                compute_spent_ns=failed_ns - started_ns,
                            )
                        )
                        self._compute_active = None
                        feedback.extend(self._fail_locked(exc, failed_ns))
                    self._deliver(feedback)
                    break
                finished_ns = self._now()
                completed = _ComputedItem(
                    item.ticket, value, started_ns, finished_ns
                )
                with self._condition:
                    self._compute_active = None
                    self._compute_completed += 1
                    if self._fatal_error is not None:
                        feedback.append(
                            self._feedback_locked(
                                item.ticket,
                                reason=TerminalReason.PIPELINE_ABORTED,
                                outcome_class=OutcomeClass.STRUCTURAL_FAILURE,
                                stage=Stage.BEFORE_PUBLICATION,
                                now_ns=finished_ns,
                                queue_wait_ns=queue_wait_ns,
                                compute_spent_ns=completed.compute_spent_ns,
                            )
                        )
                    elif (
                        finished_ns - item.ticket.capture_timestamp_ns
                        > self._config.processing_horizon_ns
                    ):
                        feedback.append(
                            self._feedback_locked(
                                item.ticket,
                                reason=TerminalReason.PROCESSING_HORIZON_EXPIRED,
                                outcome_class=OutcomeClass.EXPIRED_WORK,
                                stage=Stage.BEFORE_PUBLICATION,
                                now_ns=finished_ns,
                                queue_wait_ns=queue_wait_ns,
                                compute_spent_ns=completed.compute_spent_ns,
                            )
                        )
                    else:
                        if self._publication_pending is not None:
                            displaced = self._publication_pending
                            feedback.append(
                                self._feedback_locked(
                                    displaced.ticket,
                                    reason=TerminalReason.SUPERSEDED_PUBLICATION_PENDING,
                                    outcome_class=OutcomeClass.INTENTIONAL_FRESHNESS_DROP,
                                    stage=Stage.PENDING_PUBLICATION,
                                    now_ns=finished_ns,
                                    queue_wait_ns=displaced.queue_wait_ns,
                                    compute_spent_ns=displaced.compute_spent_ns,
                                    replacement=completed.ticket,
                                )
                            )
                        self._publication_pending = completed
                        self._publication_pending_high_water = 1
                        self._condition.notify_all()
                self._deliver(feedback)
        finally:
            with self._condition:
                self._compute_active = None
                self._compute_done = True
                self._condition.notify_all()

    def _publication_loop(self) -> None:
        feedback: list[TerminalFeedback] = []
        with self._condition:
            self._publication_owner_thread_id = threading.get_ident()
        while True:
            feedback = []
            with self._condition:
                while (
                    self._publication_pending is None
                    and not self._compute_done
                    and self._fatal_error is None
                ):
                    self._condition.wait()
                if self._publication_pending is None and (
                    self._compute_done or self._fatal_error is not None
                ):
                    break
                item = self._publication_pending
                assert item is not None
                self._publication_pending = None
                self._publication_active = item
            started_ns = self._now()
            if (
                started_ns - item.ticket.capture_timestamp_ns
                > self._config.processing_horizon_ns
            ):
                with self._condition:
                    feedback.append(
                        self._feedback_locked(
                            item.ticket,
                            reason=TerminalReason.PROCESSING_HORIZON_EXPIRED,
                            outcome_class=OutcomeClass.EXPIRED_WORK,
                            stage=Stage.BEFORE_PUBLICATION,
                            now_ns=started_ns,
                            queue_wait_ns=item.queue_wait_ns,
                            compute_spent_ns=item.compute_spent_ns,
                        )
                    )
                    self._publication_active = None
                self._deliver(feedback)
                continue
            with self._condition:
                self._publication_started += 1
                self._mark_stage_active_locked()
            try:
                self._publish(item.ticket, item.value)
            except BaseException as exc:
                failed_ns = self._now()
                with self._condition:
                    feedback.append(
                        self._feedback_locked(
                            item.ticket,
                            reason=TerminalReason.PUBLICATION_FAILED,
                            outcome_class=OutcomeClass.STRUCTURAL_FAILURE,
                            stage=Stage.PUBLICATION,
                            now_ns=failed_ns,
                            queue_wait_ns=item.queue_wait_ns,
                            compute_spent_ns=item.compute_spent_ns,
                            publication_spent_ns=failed_ns - started_ns,
                        )
                    )
                    self._publication_active = None
                    feedback.extend(self._fail_locked(exc, failed_ns))
                self._deliver(feedback)
                break
            finished_ns = self._now()
            with self._condition:
                self._publication_completed += 1
                feedback.append(
                    self._feedback_locked(
                        item.ticket,
                        reason=TerminalReason.RESULT_PUBLISHED,
                        outcome_class=OutcomeClass.PUBLICATION_SUCCESS,
                        stage=Stage.PUBLICATION,
                        now_ns=finished_ns,
                        queue_wait_ns=item.queue_wait_ns,
                        compute_spent_ns=item.compute_spent_ns,
                        publication_spent_ns=finished_ns - started_ns,
                    )
                )
                self._publication_active = None
            self._deliver(feedback)

    def close_and_join(self, *, timeout_s: float = 30.0) -> tuple[TerminalFeedback, ...]:
        _require(timeout_s > 0.0, "join timeout must be positive")
        with self._condition:
            _require(self._started, "pipeline is not started")
            self._accepting = False
            self._closing = True
            self._condition.notify_all()
            compute_thread = self._compute_thread
            publication_thread = self._publication_thread
        assert compute_thread is not None and publication_thread is not None
        deadline = time.monotonic() + timeout_s
        compute_thread.join(max(0.0, deadline - time.monotonic()))
        publication_thread.join(max(0.0, deadline - time.monotonic()))
        if compute_thread.is_alive() or publication_thread.is_alive():
            raise TimeoutError("pipeline workers did not stop within the drain window")
        with self._condition:
            if self._fatal_error is not None:
                raise PipelineWorkerError("pipeline worker failed") from self._fatal_error
            if len(self._terminal) != self._offered_frames:
                raise PipelineWorkerError(
                    "terminal accounting does not reconcile: "
                    f"offered={self._offered_frames} terminal={len(self._terminal)}"
                )
            return tuple(self._terminal.values())

    def snapshot(self) -> PipelineSnapshot:
        with self._condition:
            return PipelineSnapshot(
                policy=self._config.policy.value,
                accepting=self._accepting,
                compute_pending_depth=int(self._compute_pending is not None),
                publication_pending_depth=int(self._publication_pending is not None),
                compute_active=self._compute_active is not None,
                publication_active=self._publication_active is not None,
                offered_frames=self._offered_frames,
                compute_started=self._compute_started,
                compute_completed=self._compute_completed,
                publication_started=self._publication_started,
                publication_completed=self._publication_completed,
                terminal_frames=len(self._terminal),
                compute_pending_high_water=self._compute_pending_high_water,
                publication_pending_high_water=self._publication_pending_high_water,
                maximum_active_stage_workers=self._maximum_active_stage_workers,
                stage_overlap_observed=self._stage_overlap_observed,
                compute_owner_thread_id=self._compute_owner_thread_id,
                publication_owner_thread_id=self._publication_owner_thread_id,
                fatal_error=(
                    None
                    if self._fatal_error is None
                    else f"{type(self._fatal_error).__name__}: {self._fatal_error}"
                ),
            )

    @property
    def outcomes(self) -> tuple[TerminalFeedback, ...]:
        with self._condition:
            return tuple(self._terminal.values())

"""Bounded privileged evaluator with an honest early semantic branch."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import protocol
from .gt_evidence import read_ground_truth, read_semantic_ground_truth
from .scoring import (
    QualityInputs,
    immutable_mask,
    immutable_rows,
    require_exact_parity,
    score_localization,
    score_segmentation,
    score_serial,
)


class QualityCoordinatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalPredictionTicket:
    identity: dict[str, Any]
    frozen_carla_frame_id: int
    records: tuple[dict[str, Any], ...]
    final_mask: np.ndarray | Future[np.ndarray]
    production_mask_for_parity: np.ndarray | Future[np.ndarray] | None
    model_ready_wall_ns: int
    model_ready_monotonic_ns: int
    final_prediction_ready_wall_ns: int
    final_prediction_ready_monotonic_ns: int
    ownership_started_wall_ns: int
    ownership_completed_wall_ns: int
    evaluation_enqueued_wall_ns: int
    evaluation_enqueued_monotonic_ns: int
    upstream_timing: dict[str, Any]


@dataclass
class _EarlySegmentation:
    score_future: Future[dict[str, Any]]
    submitted_wall_ns: int
    submitted_monotonic_ns: int


class BoundedQualityCoordinator:
    """One explicit quality outcome for every submitted final prediction.

    The measured candidate ACK contains only the candidate evaluation.  A
    bounded serial reference sample runs after ACK emission and can still fail
    the cell at drain, but never inflates measured feedback latency.
    """

    def __init__(
        self,
        *,
        gt_directory: Any,
        emit_ack: Callable[[Mapping[str, Any]], None],
        record_detail: Callable[[Mapping[str, Any]], None] | None = None,
        record_validation: Callable[[Mapping[str, Any]], None] | None = None,
        match_distance_m: float = 3.0,
        queue_depth: int = 64,
        gt_timeout_s: float = 2.0,
        mode: str = "EARLY_SEMANTIC_CPU_EXACT_PLUS_FINAL_LOCALIZATION_V1",
        parity_sample_limit: int = 8,
    ) -> None:
        self.gt_directory = gt_directory
        self.emit_ack = emit_ack
        self.record_detail = record_detail or (lambda _row: None)
        self.record_validation = record_validation or (lambda _row: None)
        self.match_distance_m = float(match_distance_m)
        self.gt_timeout_s = float(gt_timeout_s)
        self.mode = str(mode)
        self.parity_sample_limit = int(parity_sample_limit)
        self._queue: "queue.Queue[FinalPredictionTicket | None]" = queue.Queue(
            maxsize=int(queue_depth)
        )
        # Two workers bound the worst-case GT wait at shutdown; queued futures
        # are cancelled rather than allowed to drain N*timeout seconds.
        self._seg_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="privileged-segmentation"
        )
        self._parity_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="privileged-quality-parity"
        )
        self._early: dict[tuple[Any, ...], _EarlySegmentation] = {}
        self._early_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._outcomes: set[tuple[Any, ...]] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="privileged-quality-evaluator", daemon=True
        )
        self._thread.start()
        self.failures: list[str] = []
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.queue_overflow = 0
        self.parity_checked = 0
        self._parity_submitted = 0
        self._parity_futures: list[Future[None]] = []

    @staticmethod
    def _key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(identity[name] for name in protocol.REQUIRED_IDENTITY)

    def submit_early_segmentation(
        self,
        identity: Mapping[str, Any],
        mask_or_future: np.ndarray | Future[np.ndarray],
    ) -> None:
        key = self._key(identity)
        owned_future: Future[np.ndarray]
        if isinstance(mask_or_future, Future):
            owned_future = mask_or_future
        else:
            owned_future = Future()
            owned_future.set_result(immutable_mask(mask_or_future))
        score_future = self._seg_pool.submit(
            self._score_early_segmentation, dict(identity), owned_future
        )
        with self._early_lock:
            if key in self._early:
                score_future.cancel()
                raise QualityCoordinatorError(f"duplicate early segmentation: {key}")
            self._early[key] = _EarlySegmentation(
                score_future=score_future,
                submitted_wall_ns=time.time_ns(),
                submitted_monotonic_ns=time.monotonic_ns(),
            )

    def _score_early_segmentation(
        self,
        identity: Mapping[str, Any],
        mask_future: Future[np.ndarray],
    ) -> dict[str, Any]:
        gt_read_started_wall_ns = time.time_ns()
        semantic_gt = read_semantic_ground_truth(
            self.gt_directory,
            expected_identity=identity,
            timeout_s=self.gt_timeout_s,
        )
        gt_read_completed_wall_ns = time.time_ns()
        predicted = immutable_mask(mask_future.result(timeout=self.gt_timeout_s))
        score_started_wall_ns = time.time_ns()
        score = score_segmentation(predicted, semantic_gt["semantic"])
        score_completed_wall_ns = time.time_ns()
        return {
            "score": score,
            "gt_ready_wall_ns": int(semantic_gt["gt_ready_wall_ns"]),
            "gt_read_started_wall_ns": gt_read_started_wall_ns,
            "gt_read_completed_wall_ns": gt_read_completed_wall_ns,
            "score_started_wall_ns": score_started_wall_ns,
            "score_completed_wall_ns": score_completed_wall_ns,
            "semantic_gt_sha256": semantic_gt["semantic_gt_sha256"],
        }

    def submit_final(
        self,
        *,
        identity: Mapping[str, Any],
        frozen_carla_frame_id: int,
        records: Sequence[Mapping[str, Any]],
        final_mask: np.ndarray | Future[np.ndarray],
        production_mask_for_parity: np.ndarray | Future[np.ndarray] | None = None,
        model_ready_wall_ns: int,
        model_ready_monotonic_ns: int,
        final_prediction_ready_wall_ns: int,
        final_prediction_ready_monotonic_ns: int,
        upstream_timing: Mapping[str, Any] | None = None,
    ) -> None:
        frame_id = int(identity["frame_id"])
        if int(frozen_carla_frame_id) != frame_id:
            raise QualityCoordinatorError("prediction frame/snapshot mismatch")
        ownership_started_wall_ns = time.time_ns()
        owned_records = immutable_rows(records)
        owned_mask = final_mask if isinstance(final_mask, Future) else immutable_mask(final_mask)
        ownership_completed_wall_ns = time.time_ns()
        enqueued_wall_ns, enqueued_mono_ns = time.time_ns(), time.monotonic_ns()
        ticket = FinalPredictionTicket(
            identity={name: identity[name] for name in protocol.REQUIRED_IDENTITY},
            frozen_carla_frame_id=frame_id,
            records=owned_records,
            final_mask=owned_mask,
            production_mask_for_parity=production_mask_for_parity,
            model_ready_wall_ns=int(model_ready_wall_ns),
            model_ready_monotonic_ns=int(model_ready_monotonic_ns),
            final_prediction_ready_wall_ns=int(final_prediction_ready_wall_ns),
            final_prediction_ready_monotonic_ns=int(final_prediction_ready_monotonic_ns),
            ownership_started_wall_ns=ownership_started_wall_ns,
            ownership_completed_wall_ns=ownership_completed_wall_ns,
            evaluation_enqueued_wall_ns=enqueued_wall_ns,
            evaluation_enqueued_monotonic_ns=enqueued_mono_ns,
            upstream_timing=dict(upstream_timing or {}),
        )
        try:
            self._queue.put_nowait(ticket)
        except queue.Full as exc:
            self.queue_overflow += 1
            self.failures.append(f"QUALITY_EVALUATION_QUEUE_OVERFLOW:{self._key(identity)}")
            self._emit_failure(ticket, "QUEUE_OVERFLOW")
            raise QualityCoordinatorError("quality evaluation queue overflow") from exc
        self.submitted += 1

    def submit_failure(
        self,
        *,
        identity: Mapping[str, Any],
        frozen_carla_frame_id: int,
        reason: str,
        model_ready_wall_ns: int | None = None,
        model_ready_monotonic_ns: int | None = None,
        final_prediction_ready_wall_ns: int | None = None,
        final_prediction_ready_monotonic_ns: int | None = None,
    ) -> None:
        """Close an eligible diagnostic ticket without inventing timestamps."""

        now_wall, now_mono = time.time_ns(), time.monotonic_ns()
        timing = {
            "clock_domains": "wall_ns_and_process_monotonic_ns_paired",
            "model_ready_wall_ns": model_ready_wall_ns,
            "model_ready_monotonic_ns": model_ready_monotonic_ns,
            "final_prediction_ready_wall_ns": final_prediction_ready_wall_ns,
            "final_prediction_ready_monotonic_ns": final_prediction_ready_monotonic_ns,
            "gt_ready_wall_ns": None,
            "evaluation_enqueued_wall_ns": None,
            "evaluation_started_wall_ns": None,
            "evaluation_completed_wall_ns": now_wall,
            "ack_emit_start_wall_ns": now_wall,
            "ack_emit_start_monotonic_ns": now_mono,
        }
        self._emit_outcome(
            identity=identity,
            frozen_carla_frame_id=frozen_carla_frame_id,
            timing=timing,
            quality=None,
            failure_reason=str(reason)[:160],
        )

    def _run(self) -> None:
        while True:
            try:
                ticket = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if ticket is None:
                self._queue.task_done()
                return
            try:
                self._evaluate(ticket)
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                self.failures.append(f"{self._key(ticket.identity)}:{reason}")
                try:
                    self._emit_failure(ticket, reason[:160])
                except QualityCoordinatorError as duplicate:
                    # A post-emission parity scheduling problem must never
                    # produce a conflicting second wire outcome.
                    self.failures.append(str(duplicate))
            finally:
                self._queue.task_done()

    def _emit_failure(self, ticket: FinalPredictionTicket, reason: str) -> None:
        now_wall, now_mono = time.time_ns(), time.monotonic_ns()
        timing = {
            "clock_domains": "wall_ns_and_process_monotonic_ns_paired",
            "model_ready_wall_ns": ticket.model_ready_wall_ns,
            "model_ready_monotonic_ns": ticket.model_ready_monotonic_ns,
            "final_prediction_ready_wall_ns": ticket.final_prediction_ready_wall_ns,
            "final_prediction_ready_monotonic_ns": ticket.final_prediction_ready_monotonic_ns,
            "gt_ready_wall_ns": None,
            "evaluation_enqueued_wall_ns": ticket.evaluation_enqueued_wall_ns,
            "evaluation_enqueued_monotonic_ns": ticket.evaluation_enqueued_monotonic_ns,
            "evaluation_started_wall_ns": None,
            "evaluation_completed_wall_ns": now_wall,
            "ack_emit_start_wall_ns": now_wall,
            "ack_emit_start_monotonic_ns": now_mono,
            "ownership_started_wall_ns": ticket.ownership_started_wall_ns,
            "ownership_completed_wall_ns": ticket.ownership_completed_wall_ns,
        }
        self._emit_outcome(
            identity=ticket.identity,
            frozen_carla_frame_id=ticket.frozen_carla_frame_id,
            timing=timing,
            quality=None,
            failure_reason=str(reason)[:160],
        )

    def _evaluate(self, ticket: FinalPredictionTicket) -> None:
        worker_dequeued_wall_ns = time.time_ns()
        worker_dequeued_mono_ns = time.monotonic_ns()
        gt_read_started_wall_ns = time.time_ns()
        gt = read_ground_truth(
            self.gt_directory,
            expected_identity=ticket.identity,
            timeout_s=self.gt_timeout_s,
        )
        gt_read_completed_wall_ns = time.time_ns()
        key = self._key(ticket.identity)
        with self._early_lock:
            early = self._early.pop(key, None)
        final_mask = immutable_mask(
            ticket.final_mask.result(timeout=self.gt_timeout_s)
            if isinstance(ticket.final_mask, Future) else ticket.final_mask
        )
        score_started_wall_ns, score_started_mono_ns = time.time_ns(), time.monotonic_ns()
        if early is None:
            segmentation_detail = {
                "score": score_segmentation(final_mask, gt["semantic"]),
                "score_started_wall_ns": score_started_wall_ns,
                "score_completed_wall_ns": time.time_ns(),
                "gt_ready_wall_ns": gt["gt_ready_wall_ns"],
                "semantic_gt_sha256": gt["semantic_gt_sha256"],
            }
        else:
            segmentation_detail = early.score_future.result(timeout=self.gt_timeout_s)
        localization_started_wall_ns = time.time_ns()
        localization = score_localization(
            ticket.records, gt["objects"], match_distance_m=self.match_distance_m
        )
        localization_completed_wall_ns = time.time_ns()
        candidate = {
            "segmentation": segmentation_detail["score"],
            "localization": localization,
        }
        completed_wall_ns, completed_mono_ns = time.time_ns(), time.monotonic_ns()
        timing = {
            "clock_domains": "wall_ns_and_process_monotonic_ns_paired",
            "model_ready_wall_ns": ticket.model_ready_wall_ns,
            "model_ready_monotonic_ns": ticket.model_ready_monotonic_ns,
            # This boundary is deliberately named honestly in the detailed row:
            # final p025 records plus their CPU serialization are ready.
            "final_prediction_ready_wall_ns": ticket.final_prediction_ready_wall_ns,
            "final_prediction_ready_monotonic_ns": ticket.final_prediction_ready_monotonic_ns,
            "final_p025_serialized_ready_wall_ns": ticket.final_prediction_ready_wall_ns,
            "gt_ready_wall_ns": int(gt["gt_ready_wall_ns"]),
            "evaluation_enqueued_wall_ns": ticket.evaluation_enqueued_wall_ns,
            "evaluation_enqueued_monotonic_ns": ticket.evaluation_enqueued_monotonic_ns,
            "evaluation_started_wall_ns": worker_dequeued_wall_ns,
            "evaluation_started_monotonic_ns": worker_dequeued_mono_ns,
            "evaluation_completed_wall_ns": completed_wall_ns,
            "evaluation_completed_monotonic_ns": completed_mono_ns,
            "ack_emit_start_wall_ns": completed_wall_ns,
            "ack_emit_start_monotonic_ns": completed_mono_ns,
            "ownership_started_wall_ns": ticket.ownership_started_wall_ns,
            "ownership_completed_wall_ns": ticket.ownership_completed_wall_ns,
            "gt_read_started_wall_ns": gt_read_started_wall_ns,
            "gt_read_completed_wall_ns": gt_read_completed_wall_ns,
            "score_started_wall_ns": score_started_wall_ns,
            "segmentation_score_started_wall_ns": segmentation_detail.get("score_started_wall_ns"),
            "segmentation_score_completed_wall_ns": segmentation_detail.get("score_completed_wall_ns"),
            "localization_score_started_wall_ns": localization_started_wall_ns,
            "localization_score_completed_wall_ns": localization_completed_wall_ns,
            "early_segmentation_branch": early is not None,
            "early_segmentation_submitted_wall_ns": (
                early.submitted_wall_ns if early is not None else None
            ),
            "semantic_gt_sha256": gt["semantic_gt_sha256"],
            "object_gt_sha256": gt["object_gt_sha256"],
            **ticket.upstream_timing,
        }
        # Emission is the final measured candidate step.  Nothing below this
        # call is allowed to emit another outcome for the same identity.
        self._emit_outcome(
            identity=ticket.identity,
            frozen_carla_frame_id=ticket.frozen_carla_frame_id,
            timing=timing,
            quality=candidate,
            failure_reason="",
        )
        if self._parity_submitted < self.parity_sample_limit:
            inputs = QualityInputs.own(
                predicted_mask=final_mask,
                ground_truth_mask=gt["semantic"],
                predictions=ticket.records,
                ground_truth_objects=gt["objects"],
                match_distance_m=self.match_distance_m,
            )
            try:
                future = self._parity_pool.submit(
                    self._validate_reference,
                    ticket.identity,
                    inputs,
                    candidate,
                    ticket.production_mask_for_parity,
                )
            except Exception as exc:
                self.failures.append(f"PARITY_SUBMIT:{type(exc).__name__}:{exc}")
            else:
                self._parity_submitted += 1
                self._parity_futures.append(future)

    def _emit_outcome(
        self,
        *, identity: Mapping[str, Any], frozen_carla_frame_id: int,
        timing: Mapping[str, Any], quality: Mapping[str, Any] | None,
        failure_reason: str,
    ) -> None:
        key = self._key(identity)
        with self._state_lock:
            if key in self._outcomes:
                raise QualityCoordinatorError(f"duplicate quality outcome: {key}")
            detail = protocol.build_detail(
                identity_fields=identity,
                frozen_carla_frame_id=frozen_carla_frame_id,
                timing=timing,
                quality=quality,
                evaluator_mode=self.mode,
                failure_reason=failure_reason,
            )
            detail_sha256 = protocol.detail_digest(detail)
            ack = protocol.build_ack(
                identity_fields=identity,
                frozen_carla_frame_id=frozen_carla_frame_id,
                timing=timing,
                quality=quality,
                evaluator_mode=self.mode,
                detail_sha256=detail_sha256,
                failure_reason=failure_reason,
            )
            # Register the full detail before its hash-bearing radio message.
            # The edge report is atomically persisted at clean close; the live
            # gate rejects a missing report or detail/hash join.
            self.record_detail({**detail, "sha256": detail_sha256})
            self.emit_ack(ack)
            self._outcomes.add(key)
            if failure_reason:
                self.failed += 1
            else:
                self.completed += 1

    def _validate_reference(
        self, identity: Mapping[str, Any], inputs: QualityInputs,
        candidate: Mapping[str, Any],
        production_mask_for_parity: np.ndarray | Future[np.ndarray] | None,
    ) -> None:
        started_wall_ns, started_mono_ns = time.time_ns(), time.monotonic_ns()
        try:
            if production_mask_for_parity is None:
                raise QualityCoordinatorError(
                    "bounded parity sample lacks the production semantic-label map"
                )
            production_mask = immutable_mask(
                production_mask_for_parity.result(timeout=self.gt_timeout_s)
                if isinstance(production_mask_for_parity, Future)
                else production_mask_for_parity
            )
            early_mask_exact = bool(
                np.array_equal(inputs.predicted_mask, production_mask)
            )
            if not early_mask_exact:
                raise QualityCoordinatorError(
                    "early semantic branch differs from production semantic labels"
                )
            production_inputs = QualityInputs.own(
                predicted_mask=production_mask,
                ground_truth_mask=inputs.ground_truth_mask,
                predictions=inputs.predictions,
                ground_truth_objects=inputs.ground_truth_objects,
                match_distance_m=inputs.match_distance_m,
            )
            reference = score_serial(production_inputs)
            require_exact_parity(reference, candidate)
            completed_wall_ns, completed_mono_ns = time.time_ns(), time.monotonic_ns()
            self.parity_checked += 1
            self.record_validation(
                {
                    "schema": "splitfusion_privileged_quality_serial_validation.v1",
                    **{name: identity[name] for name in protocol.IDENTITY_FIELDS},
                    "serial_started_wall_ns": started_wall_ns,
                    "serial_started_monotonic_ns": started_mono_ns,
                    "serial_completed_wall_ns": completed_wall_ns,
                    "serial_completed_monotonic_ns": completed_mono_ns,
                    "serial_duration_ms": (completed_mono_ns - started_mono_ns) / 1e6,
                    "reference_sha256": protocol.digest(reference),
                    "candidate_sha256": protocol.digest(candidate),
                    "early_semantic_mask_sha256": hashlib.sha256(
                        inputs.predicted_mask.tobytes()
                    ).hexdigest(),
                    "production_semantic_mask_sha256": hashlib.sha256(
                        production_mask.tobytes()
                    ).hexdigest(),
                    "early_production_mask_exact_equal": early_mask_exact,
                    "exact_parity": True,
                }
            )
        except Exception as exc:
            self.failures.append(f"SERIAL_CANDIDATE_PARITY:{type(exc).__name__}:{exc}")
            raise

    def close(self, timeout_s: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        unfinished = int(self._queue.unfinished_tasks)
        if unfinished:
            self.failures.append(f"QUALITY_EVALUATION_DRAIN_TIMEOUT:{unfinished}")
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        # Python 3.10 supports cancel_futures.  Never wait through hundreds of
        # queued GT timeouts after the registered close deadline.
        self._seg_pool.shutdown(wait=False, cancel_futures=True)
        while (
            any(not future.done() for future in self._parity_futures)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self._parity_pool.shutdown(wait=False, cancel_futures=True)
        parity_cancelled = 0
        for future in self._parity_futures:
            if not future.done():
                future.cancel()
                parity_cancelled += 1
                continue
            try:
                future.result()
            except Exception:
                pass
        with self._early_lock:
            early_missing_final = len(self._early)
            for entry in self._early.values():
                entry.score_future.cancel()
            self._early.clear()
        if early_missing_final:
            self.failures.append(f"EARLY_SEGMENTATION_WITHOUT_FINAL:{early_missing_final}")
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "queue_overflow": self.queue_overflow,
            "parity_submitted": self._parity_submitted,
            "parity_checked": self.parity_checked,
            "parity_cancelled_at_deadline": parity_cancelled,
            "early_without_final": early_missing_final,
            "unfinished_at_deadline": unfinished,
            "failures": list(self.failures),
            "worker_alive": self._thread.is_alive(),
        }

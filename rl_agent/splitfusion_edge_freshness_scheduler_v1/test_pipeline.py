from __future__ import annotations

import threading
import time
import unittest

from rl_agent.splitfusion_edge_freshness_scheduler_v1.pipeline import (
    BoundedTwoStagePipeline,
    CandidatePolicy,
    PipelineConfig,
    PipelineWorkerError,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.scheduler import (
    FrameTicket,
    OutcomeClass,
    TerminalReason,
)


MS = 1_000_000


def ticket(
    sequence: int,
    *,
    arrival_ns: int | None = None,
    capture_ns: int | None = None,
    stream_id: str = "ue-1",
) -> FrameTicket:
    arrival = time.time_ns() if arrival_ns is None else arrival_ns
    return FrameTicket(
        run_id="run",
        cell_id="cell",
        stream_id=stream_id,
        frame_id=1000 + sequence,
        sequence_id=sequence,
        action_id=71,
        capture_timestamp_ns=(
            max(0, arrival - MS) if capture_ns is None else capture_ns
        ),
        edge_arrival_timestamp_ns=arrival,
        feature_bytes=6464,
    )


def reasons(pipeline: BoundedTwoStagePipeline) -> dict[int, TerminalReason]:
    return {
        item.ticket.sequence_id: item.reason for item in pipeline.outcomes
    }


class BoundedTwoStagePipelineTest(unittest.TestCase):
    def test_active_one_then_latest_four_without_kernel_preemption(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        first_publication_started = threading.Event()
        computed: list[int] = []
        published: list[int] = []

        def compute(frame: FrameTicket, payload: int) -> int:
            computed.append(frame.sequence_id)
            if frame.sequence_id == 1:
                first_started.set()
                self.assertTrue(release_first.wait(2.0))
            elif frame.sequence_id == 4:
                self.assertTrue(first_publication_started.wait(2.0))
            return payload * 10

        def publish(frame: FrameTicket, value: int) -> None:
            published.append(frame.sequence_id)
            self.assertEqual(value, frame.sequence_id * 10)
            if frame.sequence_id == 1:
                first_publication_started.set()

        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            compute=compute,
            publish=publish,
        )
        pipeline.start()
        pipeline.offer(ticket(1), 1)
        self.assertTrue(first_started.wait(2.0))
        pipeline.offer(ticket(2), 2)
        pipeline.offer(ticket(3), 3)
        pipeline.offer(ticket(4), 4)
        release_first.set()
        outcomes = pipeline.close_and_join(timeout_s=3.0)

        self.assertEqual(computed, [1, 4])
        self.assertEqual(published, [1, 4])
        self.assertEqual(len(outcomes), 4)
        self.assertEqual(reasons(pipeline)[2], TerminalReason.SUPERSEDED_PENDING)
        self.assertEqual(reasons(pipeline)[3], TerminalReason.SUPERSEDED_PENDING)
        self.assertEqual(reasons(pipeline)[1], TerminalReason.RESULT_PUBLISHED)
        self.assertEqual(reasons(pipeline)[4], TerminalReason.RESULT_PUBLISHED)

    def test_25_ms_is_expiry_not_an_idle_worker_hold(self) -> None:
        calls: list[int] = []
        now = time.time_ns()
        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_25_MS),
            compute=lambda frame, payload: calls.append(frame.sequence_id),
            publish=lambda _frame, _value: None,
        )
        pipeline.start()
        pipeline.offer(ticket(1, arrival_ns=now - 30 * MS), None, now_ns=now)
        pipeline.close_and_join(timeout_s=2.0)
        self.assertEqual(calls, [])
        self.assertEqual(
            reasons(pipeline)[1], TerminalReason.QUEUE_WAIT_BUDGET_EXCEEDED
        )

        immediate = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_25_MS),
            compute=lambda _frame, payload: payload,
            publish=lambda _frame, _value: None,
        )
        immediate.start()
        immediate.offer(ticket(2), "value")
        immediate.close_and_join(timeout_s=2.0)
        self.assertEqual(reasons(immediate)[2], TerminalReason.RESULT_PUBLISHED)

    def test_compute_and_publication_overlap_on_distinct_owner_threads(self) -> None:
        publication_started = threading.Event()
        release_publication = threading.Event()
        second_compute_started = threading.Event()
        callback_threads: dict[str, set[int]] = {"compute": set(), "publish": set()}

        def compute(frame: FrameTicket, payload: int) -> int:
            callback_threads["compute"].add(threading.get_ident())
            if frame.sequence_id == 2:
                second_compute_started.set()
            return payload

        def publish(frame: FrameTicket, _value: int) -> None:
            callback_threads["publish"].add(threading.get_ident())
            if frame.sequence_id == 1:
                publication_started.set()
                self.assertTrue(release_publication.wait(2.0))

        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            compute=compute,
            publish=publish,
        )
        pipeline.start()
        pipeline.offer(ticket(1), 1)
        self.assertTrue(publication_started.wait(2.0))
        pipeline.offer(ticket(2), 2)
        self.assertTrue(second_compute_started.wait(2.0))
        snapshot = pipeline.snapshot()
        self.assertTrue(snapshot.stage_overlap_observed)
        self.assertEqual(snapshot.maximum_active_stage_workers, 2)
        release_publication.set()
        pipeline.close_and_join(timeout_s=3.0)
        self.assertEqual(len(callback_threads["compute"]), 1)
        self.assertEqual(len(callback_threads["publish"]), 1)
        self.assertNotEqual(
            next(iter(callback_threads["compute"])),
            next(iter(callback_threads["publish"])),
        )

    def test_predicted_horizon_keeps_useful_and_rejects_obsolete_work(self) -> None:
        now = time.time_ns()
        computed: list[int] = []
        useful = BoundedTwoStagePipeline(
            config=PipelineConfig(
                CandidatePolicy.PREDICTED_INSTALL_HORIZON,
                initial_predicted_compute_ns=100 * MS,
                initial_predicted_publication_ns=10 * MS,
                predicted_post_publication_install_ns=5 * MS,
            ),
            compute=lambda frame, payload: computed.append(frame.sequence_id),
            publish=lambda _frame, _value: None,
        )
        useful.start()
        useful.offer(
            ticket(
                1,
                arrival_ns=now - MS,
                capture_ns=now - 100 * MS,
            ),
            None,
            now_ns=now,
        )
        useful.close_and_join(timeout_s=2.0)
        self.assertEqual(computed, [1])
        self.assertEqual(reasons(useful)[1], TerminalReason.RESULT_PUBLISHED)
        self.assertEqual(useful.snapshot().prediction_admission_evaluations, 1)
        self.assertEqual(useful.snapshot().prediction_admission_rejections, 0)

        obsolete = BoundedTwoStagePipeline(
            config=PipelineConfig(
                CandidatePolicy.PREDICTED_INSTALL_HORIZON,
                initial_predicted_compute_ns=100 * MS,
                initial_predicted_publication_ns=10 * MS,
                predicted_post_publication_install_ns=5 * MS,
            ),
            compute=lambda _frame, payload: payload,
            publish=lambda _frame, _value: None,
        )
        obsolete.start()
        obsolete.offer(
            ticket(
                2,
                arrival_ns=now - MS,
                capture_ns=now - 450 * MS,
            ),
            None,
            now_ns=now,
        )
        obsolete.close_and_join(timeout_s=2.0)
        self.assertEqual(
            reasons(obsolete)[2],
            TerminalReason.PREDICTED_MAP_INSTALL_HORIZON_EXCEEDED,
        )
        self.assertEqual(obsolete.snapshot().prediction_admission_rejections, 1)

    def test_publication_pending_is_also_latest_only(self) -> None:
        publication_started = threading.Event()
        release_publication = threading.Event()
        second_computed = threading.Event()
        third_computed = threading.Event()
        published: list[int] = []

        def compute(frame: FrameTicket, payload: int) -> int:
            if frame.sequence_id == 2:
                second_computed.set()
            if frame.sequence_id == 3:
                third_computed.set()
            return payload

        def publish(frame: FrameTicket, _value: int) -> None:
            published.append(frame.sequence_id)
            if frame.sequence_id == 1:
                publication_started.set()
                self.assertTrue(release_publication.wait(2.0))

        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            compute=compute,
            publish=publish,
        )
        pipeline.start()
        pipeline.offer(ticket(1), 1)
        self.assertTrue(publication_started.wait(2.0))
        pipeline.offer(ticket(2), 2)
        self.assertTrue(second_computed.wait(2.0))
        while pipeline.snapshot().publication_pending_depth != 1:
            time.sleep(0.001)
        pipeline.offer(ticket(3), 3)
        self.assertTrue(third_computed.wait(2.0))
        deadline = time.monotonic() + 2.0
        while 2 not in reasons(pipeline) and time.monotonic() < deadline:
            time.sleep(0.001)
        release_publication.set()
        pipeline.close_and_join(timeout_s=3.0)

        self.assertEqual(published, [1, 3])
        outcome = next(
            item for item in pipeline.outcomes if item.ticket.sequence_id == 2
        )
        self.assertEqual(
            outcome.reason, TerminalReason.SUPERSEDED_PUBLICATION_PENDING
        )
        self.assertEqual(outcome.outcome_class, OutcomeClass.INTENTIONAL_FRESHNESS_DROP)
        self.assertGreaterEqual(outcome.compute_spent_ns, 0)
        self.assertEqual(outcome.replacing_sequence_id, 3)

    def test_pipeline_is_bound_to_one_run_cell_stream(self) -> None:
        gate = threading.Event()

        def compute(_frame: FrameTicket, payload: int) -> int:
            gate.wait(1.0)
            return payload

        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            compute=compute,
            publish=lambda _frame, _value: None,
        )
        pipeline.start()
        pipeline.offer(ticket(1), 1)
        with self.assertRaisesRegex(ValueError, "crosses pipeline binding"):
            pipeline.offer(ticket(2, stream_id="ue-2"), 2)
        gate.set()
        pipeline.close_and_join(timeout_s=2.0)

    def test_worker_failure_is_structural_and_propagated_after_cleanup(self) -> None:
        def fail(_frame: FrameTicket, _payload: object) -> object:
            raise RuntimeError("deliberate compute failure")

        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            compute=fail,
            publish=lambda _frame, _value: None,
        )
        pipeline.start()
        pipeline.offer(ticket(1), None)
        with self.assertRaisesRegex(PipelineWorkerError, "pipeline worker failed"):
            pipeline.close_and_join(timeout_s=2.0)
        outcome = pipeline.outcomes[0]
        self.assertEqual(outcome.reason, TerminalReason.PROCESSING_FAILED)
        self.assertEqual(outcome.outcome_class, OutcomeClass.STRUCTURAL_FAILURE)
        self.assertIn("deliberate compute failure", pipeline.snapshot().fatal_error or "")

    def test_feedback_sink_failure_is_a_pipeline_failure(self) -> None:
        def broken_sink(_feedback: object) -> None:
            raise RuntimeError("feedback channel unavailable")

        pipeline = BoundedTwoStagePipeline(
            config=PipelineConfig(CandidatePolicy.LATEST_ONLY_NO_EXPIRY),
            compute=lambda _frame, payload: payload,
            publish=lambda _frame, _value: None,
            feedback_sink=broken_sink,
        )
        pipeline.start()
        pipeline.offer(ticket(1), None)
        with self.assertRaises(PipelineWorkerError):
            pipeline.close_and_join(timeout_s=2.0)
        self.assertIn(
            "terminal feedback sink failed", pipeline.snapshot().fatal_error or ""
        )

    def test_only_registered_candidate_policies_exist(self) -> None:
        self.assertEqual(
            {policy.value for policy in CandidatePolicy},
            {
                "LATEST_ONLY_NO_EXPIRY",
                "LATEST_ONLY_25_MS",
                "PREDICTED_INSTALL_HORIZON",
            },
        )


if __name__ == "__main__":
    unittest.main()

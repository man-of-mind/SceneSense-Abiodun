from __future__ import annotations

import random
import unittest

from rl_agent.splitfusion_hybrid_sac_v1.reward_ticket_controller import (
    B_REWARD_DEADLINE_NS,
    FeedbackDisposition,
    TerminalClass,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    FORBIDDEN_POLICY_FEATURE_SUBSTRINGS,
    POLICY_FEATURE_ORDER,
    ReplayTransitionV1,
)
from rl_agent.splitfusion_hybrid_sac_v1.synthetic_contract_environment import (
    FRAME_PERIOD_NS,
    SYNTHETIC_EVIDENCE_CLASS,
    SYNTHETIC_FIXTURE_LABEL,
    AnalyticOracleFixturePolicy,
    AnalyticOutcomeProvider,
    AnalyticStateTrace,
    CyclingFixturePolicy,
    SyntheticActionChoice,
    SyntheticContractEnvironment,
    SyntheticContractError,
    SyntheticEventKind,
    SyntheticFeedbackScenario,
    SyntheticPolicyObservation,
    SyntheticTracePoint,
    counter_uniform,
)


class _GuardTrace:
    def __init__(self, *, future_delta: float = 0.0) -> None:
        self.requests = []
        self.future_delta = future_delta

    @property
    def trace_id(self) -> str:
        return f"guard-trace-{self.future_delta}"

    def observation(self, frame_index: int) -> SyntheticTracePoint:
        if self.requests and frame_index != self.requests[-1] + 1:
            raise AssertionError("environment prefetched or reread a trace point")
        self.requests.append(frame_index)
        delta = self.future_delta if frame_index >= 4 else 0.0
        return SyntheticTracePoint(
            frame_index=frame_index,
            carla_frame_id=9000 + frame_index,
            observed_ns=frame_index * FRAME_PERIOD_NS,
            camera_si_normalized=0.3 + delta,
            radar_p40=0.4,
            achieved_snr_db=15.0,
            bsr_bytes=1024,
            mcs_index=18,
        )


class _GuardPolicy:
    def __init__(self, trace: _GuardTrace) -> None:
        self.trace = trace
        self.calls = []

    def choose(self, state: SyntheticPolicyObservation):
        # At invocation the environment has requested exactly the current
        # trace point, never a future state.
        if self.trace.requests != list(range(len(self.trace.requests))):
            raise AssertionError("future trace state was consumed before action")
        values = state.as_mapping()
        self.calls.append(state.as_tuple())
        return SyntheticActionChoice(
            mode_id=(len(self.calls) - 1) % 12,
            q=0.2 + 0.5 * values["scene_camera_si_scaled"],
        )


class SyntheticContractEnvironmentTests(unittest.TestCase):
    def _run_default(self, frame_count: int = 48):
        return SyntheticContractEnvironment(
            state_trace=AnalyticStateTrace("trace-A"),
            outcome_provider=AnalyticOutcomeProvider("outcome-A"),
            fixture_seed="environment-A",
        ).run(frame_count=frame_count, policy=CyclingFixturePolicy("policy-A"))

    def test_counter_rng_is_stateless_and_bounded(self):
        a = counter_uniform("seed", "stream", 1, 2, 3)
        random.seed(1)
        for _ in range(50):
            random.random()
        b = counter_uniform("seed", "stream", 1, 2, 3)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a, 0.0)
        self.assertLess(a, 1.0)
        self.assertNotEqual(a, counter_uniform("seed", "stream", 1, 2, 4))

    def test_report_is_deterministic_and_never_a_replay_transition(self):
        random.seed(7)
        first = self._run_default()
        random.seed(99991)
        second = self._run_default()
        self.assertEqual(first.to_canonical_dict(), second.to_canonical_dict())
        self.assertEqual(first.canonical_sha256(), second.canonical_sha256())
        self.assertEqual(first.evidence_class, SYNTHETIC_EVIDENCE_CLASS)
        self.assertEqual(first.fixture_label, SYNTHETIC_FIXTURE_LABEL)
        for decision in first.decisions:
            self.assertFalse(decision.replay_transition_v1_eligible)
            self.assertNotIsInstance(decision, ReplayTransitionV1)

    def test_all_twelve_modes_are_reached_and_q_is_continuous_bounded(self):
        report = self._run_default()
        self.assertEqual(report.policy_invocations, 24)
        self.assertEqual({result.mode_id for result in report.decisions}, set(range(12)))
        self.assertTrue(all(0 <= result.q_e4 <= 9800 for result in report.decisions))
        # The fixture deliberately uses off-anchor continuous actions.
        anchors = {0, 3000, 5000, 7000, 9000, 9800}
        self.assertTrue(any(result.q_e4 not in anchors for result in report.decisions))

    def test_preferred_surface_covers_cartesian_modes_without_id_distance(self):
        # Four scene-capacity regions x three channel/quantizer regions must
        # expose every categorical family--quantizer mode exactly once.
        modes = {
            AnalyticOutcomeProvider._preferred_from_scalars(
                camera_si_normalized=scene,
                radar_p40=scene,
                achieved_snr_scaled=channel,
                mcs_scaled=channel,
            ).mode_id
            for scene in (0.125, 0.375, 0.625, 0.875)
            for channel in (1.0 / 6.0, 0.5, 5.0 / 6.0)
        }
        self.assertEqual(modes, set(range(12)))

    def test_each_decision_requests_one_reward_and_reuses_exact_action(self):
        report = self._run_default(12)
        frames = [
            event
            for event in report.events
            if event.kind is SyntheticEventKind.FRAME_ADMISSION
        ]
        grouped = {}
        for event in frames:
            grouped.setdefault(event.decision_seq, []).append(event)
        for events in grouped.values():
            self.assertTrue(events[0].reward_requested)
            self.assertTrue(all(not event.reward_requested for event in events[1:]))
            self.assertEqual(len({event.action_sha256 for event in events}), 1)
        self.assertTrue(all(result.realized_duration_d >= 2 for result in report.decisions))

    def test_early_duplicate_timeout_late_and_inclusive_boundary(self):
        provider = AnalyticOutcomeProvider(
            "event-scenarios",
            scenario_by_decision={
                0: SyntheticFeedbackScenario.EARLY_WITH_DUPLICATE,
                1: SyntheticFeedbackScenario.LATE_AFTER_TIMEOUT,
                2: SyntheticFeedbackScenario.EXACT_DEADLINE,
            },
        )
        report = SyntheticContractEnvironment(
            state_trace=AnalyticStateTrace("event-trace"),
            outcome_provider=provider,
            fixture_seed="event-environment",
        ).run(frame_count=8, policy=AnalyticOracleFixturePolicy())

        self.assertEqual(
            [event.observed_ns for event in report.events],
            sorted(event.observed_ns for event in report.events),
        )

        by_decision = {item.decision_seq: item for item in report.decisions}
        self.assertEqual(by_decision[0].realized_duration_d, 2)
        self.assertEqual(
            by_decision[0].terminal_class, TerminalClass.REWARD_FINAL_EXACT.value
        )
        self.assertEqual(
            by_decision[1].terminal_class, TerminalClass.FEEDBACK_TIMEOUT.value
        )
        # Inclusive B means a frame at exactly +200 ms is still governed by
        # the old decision; expiry occurs at B+1 ns, hence d=3 at 10 Hz.
        self.assertEqual(by_decision[1].realized_duration_d, 3)
        self.assertIsNone(by_decision[1].feedback_latency_ns)
        self.assertEqual(
            by_decision[2].terminal_class, TerminalClass.REWARD_FINAL_EXACT.value
        )
        self.assertEqual(by_decision[2].feedback_latency_ns, B_REWARD_DEADLINE_NS)
        self.assertEqual(by_decision[2].realized_duration_d, 2)

        feedback_dispositions = [
            event.disposition
            for event in report.events
            if event.kind is SyntheticEventKind.FEEDBACK_RECEIPT
        ]
        self.assertGreaterEqual(
            feedback_dispositions.count(FeedbackDisposition.DUPLICATE_IGNORED.value),
            2,
        )
        self.assertIn(FeedbackDisposition.LATE_ORPHAN.value, feedback_dispositions)

        # At the exact deadline, queued feedback is processed before the frame
        # at the same timestamp; otherwise the inclusive contract is ambiguous.
        exact_feedback_index = next(
            event.event_index
            for event in report.events
            if event.decision_seq == 2
            and event.kind is SyntheticEventKind.FEEDBACK_RECEIPT
            and event.disposition == FeedbackDisposition.ACCEPTED_RESOLVED.value
        )
        frame_at_same_time = next(
            event.event_index
            for event in report.events
            if event.kind is SyntheticEventKind.FRAME_ADMISSION
            and event.observed_ns
            == next(
                candidate.observed_ns
                for candidate in report.events
                if candidate.event_index == exact_feedback_index
            )
        )
        self.assertLess(exact_feedback_index, frame_at_same_time)

    def test_no_feedback_times_out_without_fabricating_a_reward(self):
        report = SyntheticContractEnvironment(
            state_trace=AnalyticStateTrace("no-feedback-trace"),
            outcome_provider=AnalyticOutcomeProvider(
                "no-feedback-outcome",
                scenario_by_decision={
                    0: SyntheticFeedbackScenario.NO_FEEDBACK_TIMEOUT
                },
            ),
            fixture_seed="no-feedback-environment",
        ).run(frame_count=4, policy=AnalyticOracleFixturePolicy())
        first = next(result for result in report.decisions if result.decision_seq == 0)
        self.assertEqual(first.terminal_class, TerminalClass.FEEDBACK_TIMEOUT.value)
        self.assertIsNone(first.feedback_latency_ns)
        self.assertFalse(first.replay_transition_v1_eligible)

    def test_state_trace_is_consumed_sequentially_without_future_leakage(self):
        trace_a = _GuardTrace(future_delta=0.0)
        policy_a = _GuardPolicy(trace_a)
        report_a = SyntheticContractEnvironment(
            state_trace=trace_a,
            outcome_provider=AnalyticOutcomeProvider("guard-outcome"),
            fixture_seed="guard-environment-a",
        ).run(frame_count=8, policy=policy_a)
        self.assertEqual(trace_a.requests, list(range(8)))

        trace_b = _GuardTrace(future_delta=0.4)
        policy_b = _GuardPolicy(trace_b)
        report_b = SyntheticContractEnvironment(
            state_trace=trace_b,
            outcome_provider=AnalyticOutcomeProvider("guard-outcome"),
            fixture_seed="guard-environment-b",
        ).run(frame_count=8, policy=policy_b)
        self.assertEqual(trace_b.requests, list(range(8)))

        # The two traces are identical before frame 4.  Actions opened from
        # that prefix must therefore be identical even though the suffixes
        # differ; future state cannot affect an earlier invocation.
        prefix_a = [
            event.action_sha256
            for event in report_a.events
            if event.kind is SyntheticEventKind.FRAME_ADMISSION
            and event.reward_requested
            and event.observed_ns < 4 * FRAME_PERIOD_NS
        ]
        prefix_b = [
            event.action_sha256
            for event in report_b.events
            if event.kind is SyntheticEventKind.FRAME_ADMISSION
            and event.reward_requested
            and event.observed_ns < 4 * FRAME_PERIOD_NS
        ]
        self.assertEqual(prefix_a, prefix_b)

    def test_policy_choice_bounds_fail_closed(self):
        with self.assertRaises(SyntheticContractError):
            SyntheticActionChoice(mode_id=12, q=0.5)
        with self.assertRaises(SyntheticContractError):
            SyntheticActionChoice(mode_id=0, q=-0.01)
        with self.assertRaises(SyntheticContractError):
            SyntheticActionChoice(mode_id=0, q=0.981)
        with self.assertRaises(SyntheticContractError):
            SyntheticActionChoice(mode_id=0, q=float("nan"))

    def test_policy_receives_only_frozen_allowed_feature_values(self):
        class InspectingPolicy:
            def __init__(self):
                self.observations = []

            def choose(self, state):
                self.observations.append(state)
                return SyntheticActionChoice(mode_id=0, q=0.5)

        policy = InspectingPolicy()
        SyntheticContractEnvironment(
            state_trace=AnalyticStateTrace("feature-boundary-trace"),
            outcome_provider=AnalyticOutcomeProvider("feature-boundary-outcome"),
            fixture_seed="feature-boundary-environment",
        ).run(frame_count=4, policy=policy)
        self.assertGreaterEqual(len(policy.observations), 1)
        for observation in policy.observations:
            self.assertIsInstance(observation, SyntheticPolicyObservation)
            self.assertEqual(tuple(observation.as_mapping()), POLICY_FEATURE_ORDER)
            for forbidden_attribute in (
                "frame_index",
                "carla_frame_id",
                "observed_ns",
                "decision_seq",
                "tensor_seq",
            ):
                self.assertFalse(hasattr(observation, forbidden_attribute))
        for name in POLICY_FEATURE_ORDER:
            self.assertFalse(
                any(forbidden in name for forbidden in FORBIDDEN_POLICY_FEATURE_SUBSTRINGS)
            )

    def test_previous_action_and_outcome_are_projected_without_identifiers(self):
        class RecordingPolicy:
            def __init__(self):
                self.states = []

            def choose(self, state):
                self.states.append(state.as_mapping())
                return SyntheticActionChoice(mode_id=3, q=0.4)

        policy = RecordingPolicy()
        SyntheticContractEnvironment(
            state_trace=AnalyticStateTrace("previous-feature-trace"),
            outcome_provider=AnalyticOutcomeProvider("previous-feature-outcome"),
            fixture_seed="previous-feature-environment",
        ).run(frame_count=6, policy=policy)
        first = policy.states[0]
        second = policy.states[1]
        self.assertEqual(first["prev_present_mask"], 0.0)
        self.assertEqual(second["prev_present_mask"], 1.0)
        self.assertEqual(second["prev_joint_mode_onehot_03"], 1.0)
        self.assertEqual(second["prev_terminal_onehot_exact"], 1.0)
        self.assertNotIn("frame_index", second)
        self.assertNotIn("carla_frame_id", second)
        self.assertNotIn("observed_ns", second)

    def test_action_path_failure_is_visible_but_has_no_quality_or_latency(self):
        class RecordingPolicy:
            def __init__(self):
                self.states = []

            def choose(self, state):
                self.states.append(state.as_mapping())
                return SyntheticActionChoice(mode_id=2, q=0.5)

        policy = RecordingPolicy()
        SyntheticContractEnvironment(
            state_trace=AnalyticStateTrace("failure-feature-trace"),
            outcome_provider=AnalyticOutcomeProvider(
                "failure-feature-outcome",
                scenario_by_decision={
                    0: SyntheticFeedbackScenario.ACTION_PATH_FAILURE
                },
            ),
            fixture_seed="failure-feature-environment",
        ).run(frame_count=5, policy=policy)
        after_failure = policy.states[1]
        self.assertEqual(
            after_failure["prev_terminal_onehot_action_path_failure"], 1.0
        )
        self.assertEqual(after_failure["prev_quality_valid_mask"], 0.0)
        self.assertEqual(after_failure["prev_latency_valid_mask"], 0.0)


if __name__ == "__main__":
    unittest.main()

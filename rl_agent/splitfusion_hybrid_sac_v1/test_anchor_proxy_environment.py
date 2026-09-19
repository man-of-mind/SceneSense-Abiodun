"""Focused tests for the measured-anchor proxy qualification boundary."""

from __future__ import annotations

import math
import unittest

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    CATALOG_SHA256,
    EXPECTED_MODE_COUNT,
)
from rl_agent.splitfusion_hybrid_sac_v1.anchor_proxy_environment import (
    CONFIRMED_FAILURE_TERMINALS,
    LOAO_METHOD,
    LOAO_SPEC_SHA256,
    LOAO_THRESHOLDS,
    PROFILE_DESIGN_SHA256,
    PROXY_EVIDENCE_CLASS,
    PROXY_USE_RESTRICTION,
    SUPERSEDED_CENSORED_TERMINALS,
    TARGET_SNR_TRACE_SHA256,
    TARGET_SNR_VALUE_SEMANTICS,
    ContinuousProxyBlockedError,
    MeasuredAnchorProxyEnvironment,
    ProxyExtrapolationError,
    ProxyOutcomeGroup,
    ProxyReplayForbiddenError,
    ProxyRewardConfig,
    load_default_anchor_proxy,
)
from rl_agent.splitfusion_hybrid_sac_v1.anchor_store import (
    ACTION_SUMMARY_SHA256,
    EXPECTED_CELL_COUNT,
    NETWORK_PROFILE_ORDER,
    PROFILE_LATENCY_SHA256,
    REGISTERED_Q_ANCHORS_E4,
)


class AnchorProxyEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proxy = load_default_anchor_proxy()

    def test_all_evidence_hashes_are_bound(self) -> None:
        binding = self.proxy.binding
        self.assertEqual(binding.catalog_sha256, CATALOG_SHA256)
        self.assertEqual(binding.action_summary_sha256, ACTION_SUMMARY_SHA256)
        self.assertEqual(binding.profile_latency_sha256, PROFILE_LATENCY_SHA256)
        self.assertEqual(binding.profile_design_sha256, PROFILE_DESIGN_SHA256)
        self.assertEqual(binding.target_snr_trace_sha256, TARGET_SNR_TRACE_SHA256)
        self.assertEqual(binding.loao_spec_sha256, LOAO_SPEC_SHA256)
        self.assertEqual(binding.anchor_store_sha256, self.proxy.store.canonical_sha256())
        self.assertEqual(binding.evidence_class, PROXY_EVIDENCE_CLASS)

    def test_inventory_is_twelve_modes_by_six_anchors_and_288_cells(self) -> None:
        self.assertEqual(self.proxy.store.anchor_count, 72)
        self.assertEqual(self.proxy.store.cell_count, EXPECTED_CELL_COUNT)
        self.assertEqual(self.proxy.store.contract.mode_count, EXPECTED_MODE_COUNT)
        observed_action_ids = set()
        for mode_id in range(EXPECTED_MODE_COUNT):
            records = self.proxy.store.records_for_mode(mode_id)
            self.assertEqual(len(records), 6)
            self.assertEqual(
                tuple(record.quality.q_e4 for record in records),
                REGISTERED_Q_ANCHORS_E4,
            )
            observed_action_ids.update(record.action_id for record in records)
        self.assertEqual(observed_action_ids, set(range(72)))

    def test_target_snr_traces_are_privileged_design_targets(self) -> None:
        traces = self.proxy.traces
        self.assertIn("NOT_MEASURED_ACHIEVED_OAI_SNR", traces.claim_boundary)
        for profile in NETWORK_PROFILE_ORDER:
            points = traces.points_by_profile[profile]
            self.assertEqual(len(points), 4200)
            self.assertEqual(tuple(point.step_index for point in points), tuple(range(4200)))
            for point in (points[0], points[1337], points[-1]):
                self.assertEqual(point.value_semantics, TARGET_SNR_VALUE_SEMANTICS)
                self.assertTrue(5.5 <= point.target_snr_db <= 24.5)

    def test_loao_gate_is_preregistered_and_blocks_continuous_q(self) -> None:
        report = self.proxy.loao
        expected = EXPECTED_MODE_COUNT * 4 * (1 + len(NETWORK_PROFILE_ORDER) * 4)
        self.assertEqual(report.evaluated_count, expected)
        self.assertEqual(report.evaluated_count, 816)
        self.assertEqual(report.failed_count, 198)
        self.assertEqual(report.missing_support_count, 48)
        self.assertEqual(report.method, LOAO_METHOD)
        self.assertEqual(report.spec_sha256, LOAO_SPEC_SHA256)
        self.assertEqual(dict(report.thresholds), dict(LOAO_THRESHOLDS))
        self.assertEqual(
            dict(report.failure_count_by_metric),
            {
                "combined_quality_abs": 21,
                "payload_log_ratio_abs": 0,
                "reassembly_rate_abs": 56,
                "admission_rate_abs": 60,
                "sensor_model_ready_p50_relative": 61,
            },
        )
        self.assertFalse(report.continuous_q_qualified)
        self.assertFalse(self.proxy.continuous_q_enabled)
        self.assertGreater(report.failed_count, 0)
        self.assertGreater(report.missing_support_count, 0)
        self.assertTrue(
            any(
                item.reason == "ERROR_EXCEEDS_THRESHOLD" for item in report.records
            )
        )
        self.assertTrue(
            any(
                item.reason == "MISSING_CONDITIONAL_SUPPORT"
                for item in report.records
            )
        )
        # Pin the scientifically material reasons rather than just a boolean.
        self.assertGreater(
            report.max_error_by_metric["reassembly_rate_abs"],
            LOAO_THRESHOLDS["reassembly_rate_abs"],
        )
        self.assertGreater(
            report.max_error_by_metric["admission_rate_abs"],
            LOAO_THRESHOLDS["admission_rate_abs"],
        )
        self.assertGreater(
            report.max_error_by_metric["sensor_model_ready_p50_relative"],
            LOAO_THRESHOLDS["sensor_model_ready_p50_relative"],
        )

    def test_non_anchor_is_refused_and_extrapolation_is_separate(self) -> None:
        with self.assertRaises(ContinuousProxyBlockedError):
            self.proxy.interpolate(0, 4000, "FAVORABLE_STABLE")
        with self.assertRaises(ProxyExtrapolationError):
            self.proxy.interpolate(0, -1, "FAVORABLE_STABLE")
        with self.assertRaises(ProxyExtrapolationError):
            self.proxy.interpolate(0, 9801, "FAVORABLE_STABLE")

    def test_every_exact_action_has_bounded_proxy_coordinates(self) -> None:
        for profile in NETWORK_PROFILE_ORDER:
            for action_id in range(72):
                estimate = self.proxy.exact_action(action_id, profile)
                self.assertEqual(estimate.action_id, action_id)
                self.assertEqual(estimate.evidence_class, PROXY_EVIDENCE_CLASS)
                self.assertIn(estimate.q_e4, REGISTERED_Q_ANCHORS_E4)
                self.assertTrue(0.0 <= estimate.combined_quality <= 1.0)
                self.assertGreater(estimate.payload_bytes, 0.0)
                self.assertTrue(0.0 <= estimate.reassembly_rate <= 1.0)
                self.assertTrue(0.0 <= estimate.admission_rate <= 1.0)
                if estimate.sensor_model_ready_p50_ms is not None:
                    self.assertGreaterEqual(estimate.sensor_model_ready_p50_ms, 0.0)

    def test_policy_observation_has_no_profile_or_target_trace_leak(self) -> None:
        for profile in NETWORK_PROFILE_ORDER:
            environment = MeasuredAnchorProxyEnvironment(
                self.proxy, profile, seed="no-leak", horizon_steps=2
            )
            observation = environment.reset()
            view = observation.policy_dict()
            serialized_names = " ".join(view).lower()
            self.assertNotIn("profile", serialized_names)
            self.assertNotIn("target_snr", serialized_names)
            self.assertIsNone(view["achieved_snr_db"])
            self.assertFalse(view["achieved_snr_available"])
            self.assertIsNone(view["bsr_bytes"])
            self.assertFalse(view["bsr_available"])
            self.assertIsNone(view["mcs_index"])
            self.assertFalse(view["mcs_available"])
            hidden = environment.hidden_context()
            self.assertEqual(hidden.network_profile, profile)
            self.assertFalse(hidden.target_snr_is_achieved_measurement)
            self.assertEqual(
                hidden.target_snr_design.value_semantics,
                TARGET_SNR_VALUE_SEMANTICS,
            )

    def test_sequential_proxy_is_deterministic_and_updates_only_prior_view(self) -> None:
        left = MeasuredAnchorProxyEnvironment(
            self.proxy, "FADE_RECOVERY", seed="repeatable", horizon_steps=12
        )
        right = MeasuredAnchorProxyEnvironment(
            self.proxy, "FADE_RECOVERY", seed="repeatable", horizon_steps=12
        )
        actions = (0, 15, 30, 50, 71, 66, 68, 2, 40, 55, 69, 12)
        left_transitions = tuple(left.step(action) for action in actions)
        right_transitions = tuple(right.step(action) for action in actions)
        self.assertEqual(left_transitions, right_transitions)
        self.assertIsNone(left_transitions[0].observation.previous_action_id)
        for previous, current in zip(left_transitions, left_transitions[1:]):
            self.assertEqual(current.observation.previous_action_id, previous.action_id)
            self.assertEqual(current.observation.previous_reward, previous.outcome.reward)
        self.assertTrue(left_transitions[-1].done)

    def test_reward_config_and_loao_rule_are_bound_into_environment_identity(self) -> None:
        default_config = ProxyRewardConfig()
        same_config = ProxyRewardConfig(
            quality_weight=1.0,
            latency_weight=0.25,
            deadline_budget_ms=200.0,
            timeout_or_failure_penalty=-1.0,
            superseded_censored_penalty=-0.5,
        )
        changed_config = ProxyRewardConfig(latency_weight=0.50)
        self.assertEqual(default_config.to_canonical_dict(), same_config.to_canonical_dict())
        self.assertEqual(default_config.canonical_sha256(), same_config.canonical_sha256())
        self.assertNotEqual(
            default_config.canonical_sha256(), changed_config.canonical_sha256()
        )

        first = MeasuredAnchorProxyEnvironment(
            self.proxy,
            "FAVORABLE_STABLE",
            seed="reward-identity",
            horizon_steps=1,
            reward_config=default_config,
        )
        repeated = MeasuredAnchorProxyEnvironment(
            self.proxy,
            "FAVORABLE_STABLE",
            seed="reward-identity",
            horizon_steps=1,
            reward_config=same_config,
        )
        changed = MeasuredAnchorProxyEnvironment(
            self.proxy,
            "FAVORABLE_STABLE",
            seed="reward-identity",
            horizon_steps=1,
            reward_config=changed_config,
        )
        self.assertEqual(
            first.environment_identity_sha256, repeated.environment_identity_sha256
        )
        self.assertNotEqual(
            first.environment_identity_sha256, changed.environment_identity_sha256
        )
        first_transition = first.step(69)
        repeated_transition = repeated.step(69)
        changed_transition = changed.step(69)
        self.assertEqual(first_transition, repeated_transition)
        # Reward parameters do not alter the sampled terminal/latency; they
        # alter only the registered reward and the environment identity.
        self.assertEqual(
            first_transition.outcome.source_terminal,
            changed_transition.outcome.source_terminal,
        )
        self.assertEqual(
            first_transition.outcome.latency_ms,
            changed_transition.outcome.latency_ms,
        )
        self.assertNotEqual(
            first_transition.proxy_sequence_id,
            changed_transition.proxy_sequence_id,
        )
        self.assertNotEqual(
            first_transition.outcome.reward,
            changed_transition.outcome.reward,
        )

    def test_outcome_keeps_failure_and_censoring_distinct(self) -> None:
        environment = MeasuredAnchorProxyEnvironment(
            self.proxy, "ADVERSE_STABLE", seed="classification", horizon_steps=600
        )
        groups = set()
        for index in range(600):
            transition = environment.step(index % 72)
            outcome = transition.outcome
            groups.add(outcome.group)
            self.assertTrue(math.isfinite(outcome.reward))
            if outcome.right_censored:
                self.assertIsNone(outcome.latency_ms)
            if outcome.source_terminal in CONFIRMED_FAILURE_TERMINALS:
                self.assertTrue(outcome.confirmed_failure)
                self.assertEqual(
                    outcome.group,
                    ProxyOutcomeGroup.CONFIRMED_FAILURE_RIGHT_CENSORED,
                )
            elif outcome.source_terminal in SUPERSEDED_CENSORED_TERMINALS:
                self.assertFalse(outcome.confirmed_failure)
                self.assertEqual(
                    outcome.group,
                    ProxyOutcomeGroup.SUPERSEDED_RIGHT_CENSORED,
                )
            else:
                self.assertEqual(outcome.source_terminal, "terminal_result_published")
                self.assertTrue(outcome.published)
        self.assertIn(ProxyOutcomeGroup.CONFIRMED_FAILURE_RIGHT_CENSORED, groups)
        self.assertIn(ProxyOutcomeGroup.PUBLISHED_WITH_LATENCY, groups)

    def test_proxy_transition_is_never_replay_admissible(self) -> None:
        environment = MeasuredAnchorProxyEnvironment(
            self.proxy, "FAVORABLE_STABLE", seed="boundary", horizon_steps=1
        )
        transition = environment.step(66)
        self.assertEqual(transition.evidence_class, PROXY_EVIDENCE_CLASS)
        self.assertEqual(transition.use_restriction, PROXY_USE_RESTRICTION)
        self.assertFalse(transition.replay_admissible)
        with self.assertRaises(ProxyReplayForbiddenError):
            transition.as_replay_transition()

    def test_expected_reward_surface_is_nondegenerate(self) -> None:
        reports = []
        for profile in NETWORK_PROFILE_ORDER:
            environment = MeasuredAnchorProxyEnvironment(
                self.proxy, profile, seed="diagnostic", horizon_steps=2
            )
            report = environment.nondegeneracy_report()
            reports.append(report)
            self.assertTrue(report.nondegenerate)
            self.assertEqual(report.action_count, 72)
            self.assertGreater(report.unique_expected_rewards, 1)
            self.assertTrue(0 <= report.best_action_id < 72)
            self.assertTrue(math.isfinite(report.best_expected_reward))
        # Profile-conditioned aggregate evidence changes values even though the
        # profile label is never exposed to the policy.
        self.assertGreater(len({round(item.reward_max, 9) for item in reports}), 1)


if __name__ == "__main__":
    unittest.main()

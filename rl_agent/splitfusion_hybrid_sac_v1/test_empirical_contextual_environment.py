"""Offline CPU tests for the D1 empirical contextual contract/environment."""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import FrozenInstanceError, replace

from . import empirical_contextual_contract as contract
from . import empirical_contextual_environment as environment
from . import empirical_radio_context as radio
from .action_contract import CATALOG_SHA256
from .state_reward_transition_contract import (
    FORBIDDEN_POLICY_FEATURE_SUBSTRINGS,
    POLICY_FEATURE_ORDER,
)


class EmpiricalContextualContractTest(unittest.TestCase):
    def test_utility_overlay_is_separate_exact_and_hash_bound(self) -> None:
        spec = contract.PILOT_UTILITY_SPEC
        self.assertEqual(spec.quality_component, "q_perc")
        self.assertEqual(spec.quality_weight, 1.0)
        self.assertEqual(spec.latency_weight, 0.25)
        self.assertEqual(spec.deadline_ms, 200.0)
        self.assertEqual(spec.service_non_admission_utility, -1.0)
        self.assertEqual(spec.gamma, 1.0)
        self.assertEqual(contract.fixed_stage_latency_ms(), 113.0)
        self.assertEqual(
            spec.canonical_sha256(), contract.PILOT_UTILITY_SPEC_SHA256
        )
        self.assertIn("SEPARATE", spec.scalar_provenance_status)
        with self.assertRaises(ValueError):
            replace(spec, latency_weight=0.0)

    def test_action_support_is_exact_inclusive_and_never_projects(self) -> None:
        for mode_id, (lower, upper) in enumerate(
            contract.MODELED_SMOKE_SUPPORT.mode_q_e4_bounds
        ):
            self.assertEqual(
                contract.require_supported_action(mode_id, lower).q_e4, lower
            )
            self.assertEqual(
                contract.require_supported_action(mode_id, upper).q_e4, upper
            )
            for invalid in (lower - 1, upper + 1, float(lower), True):
                with self.assertRaises(contract.ActionSupportError):
                    contract.require_supported_action(mode_id, invalid)

    def test_expected_utility_formula(self) -> None:
        observed = contract.PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=0.8,
            q_perc=0.7,
            latency_proxy_ms=220.0,
        )
        expected = 0.8 * (0.7 - 0.25 * 220.0 / 200.0) + 0.2 * -1.0
        self.assertAlmostEqual(observed, expected)

    def test_fit_context_refuses_nonfinite_empirical_values(self) -> None:
        base = dict(
            sample_id="fit-sample",
            episode_id="fit-episode",
            frame_id=1,
            camera_si=100.0,
            radar_p40=0.5,
            sampling_weight=1.0,
            selection_rank_within_fit=0,
        )
        for field_name in ("camera_si", "radar_p40", "sampling_weight"):
            candidate = dict(base)
            candidate[field_name] = float("nan")
            with self.assertRaises(environment.EmpiricalEnvironmentError):
                environment.FitTrainingContextV1(**candidate)


class EmpiricalRadioContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = radio.OaiRadioCalibrationStoreV1.load_registered()

    def test_exact_inventory_and_calibration_ranges(self) -> None:
        self.assertEqual(self.store.source_sha256, radio.CALIBRATION_SHA256)
        self.assertEqual(len(self.store.rows), 399)
        self.assertEqual(
            dict(self.store.profile_counts),
            {
                "FAVORABLE_STABLE": 100,
                "MID_VARIABLE": 100,
                "ADVERSE_STABLE": 99,
                "FADE_RECOVERY": 100,
            },
        )
        self.assertEqual(self.store.achieved_snr_range_db, (6.0, 23.5))
        self.assertEqual(self.store.cross_profile_aliased_row_count, 140)

    def test_same_row_joint_values_and_genesis_bsr(self) -> None:
        sampler = radio.RadioContextSamplerV1(self.store, seed=91)
        draw = sampler.sample(
            observed_ns=123, control_session_id="session-for-radio-test"
        )
        row = next(
            item
            for item in self.store.rows
            if item.csv_row_number == draw.hidden_csv_row_number
        )
        self.assertEqual(row.network_profile, draw.hidden_profile)
        self.assertEqual(
            draw.observation.achieved_snr_db,
            row.achieved_pusch_snr_median_db,
        )
        self.assertEqual(draw.mcs_median, row.mcs_median)
        self.assertIn(draw.rounded_mcs_index, {math.floor(row.mcs_median), math.ceil(row.mcs_median)})
        self.assertEqual(draw.observation.bsr_report.lcg_bytes, (0,) * 8)
        self.assertEqual(draw.observation.bsr_report.valid_mask, (True,) * 8)
        self.assertIn("TRUE_ONE_STEP_EPISODE_GENESIS", draw.genesis_bsr_justification)

    def test_half_rounding_is_seeded_and_unbiased(self) -> None:
        rng = random.Random(90210)
        values = [radio.round_mcs_median_unbiased(9.5, rng)[0] for _ in range(10_000)]
        upper = sum(value == 10 for value in values)
        self.assertGreater(upper, 4_700)
        self.assertLess(upper, 5_300)
        self.assertEqual(
            radio.round_mcs_median_unbiased(28.0, random.Random(1)),
            (28, "EXACT_INTEGER_NO_ROUNDING"),
        )

    def test_rng_streams_are_independent_and_checkpointable(self) -> None:
        first = radio.RadioContextSamplerV1(self.store, seed=17)
        state = first.state_dict()
        second = radio.RadioContextSamplerV1(self.store, seed=17)
        changed_rounding = replace(
            state, rounding_rng_state=random.Random(999).getstate()
        )
        second.load_state_dict(changed_rounding)
        a = first.sample(observed_ns=1, control_session_id="radio-stream-test")
        b = second.sample(observed_ns=1, control_session_id="radio-stream-test")
        self.assertEqual(a.hidden_profile, b.hidden_profile)
        self.assertEqual(a.hidden_csv_row_number, b.hidden_csv_row_number)

        checkpoint = first.state_dict()
        expected = first.sample(observed_ns=2, control_session_id="radio-stream-test")
        restored = radio.RadioContextSamplerV1(self.store, seed=17)
        restored.load_state_dict(checkpoint)
        actual = restored.sample(observed_ns=2, control_session_id="radio-stream-test")
        self.assertEqual(expected, actual)

        before_bad_restore = restored.state_dict()
        malformed = replace(before_bad_restore, row_rng_state=("bad",))
        with self.assertRaises(radio.RadioCalibrationError):
            restored.load_state_dict(malformed)
        self.assertEqual(restored.state_dict(), before_bad_restore)


class EmpiricalOneStepEnvironmentIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = environment.load_registered_d1_environment(seed=20260921)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def _episode(self, mode_id: int = 0, q_e4: int = 8812):
        observation = self.env.reset()
        result = self.env.step(contract.require_supported_action(mode_id, q_e4))
        return observation, result

    def test_preflight_is_exhaustive_and_scientifically_scoped(self) -> None:
        report = self.env.preflight
        self.assertEqual(report.fit_scene_count, 512)
        self.assertEqual(report.rewardable_fit_context_count, 476)
        self.assertEqual(report.excluded_action_independent_invalid_count, 36)
        self.assertTrue(report.invalidity_is_action_independent)
        self.assertEqual(report.supported_endpoint_query_count, 476 * 12 * 2)
        self.assertEqual(
            report.exhaustive_network_endpoint_profile_query_count,
            476 * 12 * 2 * 4,
        )
        self.assertEqual(report.network_latency_interior_support_hole_count, 0)
        self.assertEqual(
            report.surface_qualification_report_sha256,
            contract.SURFACE_QUALIFICATION_REPORT_SHA256,
        )
        self.assertEqual(report.surface_qualification_overall_status, "FAIL")
        self.assertTrue(report.surface_q_perc_fit_held_qualified)
        self.assertFalse(report.surface_q_seg_fit_held_qualified)
        self.assertTrue(report.surface_held_payload_qualified)
        self.assertFalse(report.surface_full_component_qualified)
        self.assertTrue(report.expected_reward_nondegenerate)
        self.assertLess(*report.expected_reward_range)
        self.assertGreater(report.modeled_budget_miss_counts_p50_p95_p99[0], 0)
        self.assertEqual(
            report.environment_binding_sha256,
            self.env.binding.canonical_sha256(),
        )

    def test_binding_covers_catalog_qualification_rng_and_implementations(self) -> None:
        binding = self.env.binding
        self.assertEqual(binding.action_catalog_sha256, CATALOG_SHA256)
        self.assertEqual(
            binding.surface_qualification_report_sha256,
            contract.SURFACE_QUALIFICATION_REPORT_SHA256,
        )
        self.assertEqual(
            binding.profile_order_rng_contract_sha256,
            contract.PROFILE_ORDER_RNG_CONTRACT_SHA256,
        )
        self.assertEqual(len(binding.implementation_bundle_sha256), 64)
        with self.assertRaises(ValueError):
            replace(binding, action_catalog_sha256="0" * 64)
        with self.assertRaises(environment.EmpiricalEnvironmentError):
            replace(self.env.preflight, status="FORGED_GO")

    def test_exact_31_feature_genesis_and_no_hidden_identity(self) -> None:
        observation, _result = self._episode()
        self.assertEqual(observation.policy_feature_order, POLICY_FEATURE_ORDER)
        self.assertEqual(len(observation.values), 31)
        named = observation.as_mapping()
        self.assertEqual(named["radio_bsr_log1p_scaled"], 0.0)
        for name in (
            "freshness_scene_normalized",
            "freshness_snr_normalized",
            "freshness_bsr_normalized",
            "freshness_mcs_normalized",
        ):
            self.assertEqual(named[name], 0.0)
        self.assertTrue(all(value == 0.0 for value in observation.values[9:]))
        serialized_names = " ".join(named).lower()
        for forbidden in FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
            self.assertNotIn(forbidden, serialized_names)
        for hidden in ("target", "trace", "sample_id", "profile"):
            self.assertNotIn(hidden, serialized_names)
        with self.assertRaises(environment.EmpiricalEnvironmentError):
            replace(observation, values=observation.values[:-1])

    def test_reward_is_expected_utility_with_full_latency_diagnostics(self) -> None:
        _observation, result = self._episode(mode_id=11, q_e4=0)
        outcome = result.policy
        self.assertEqual(outcome.status, "MODELED_EXPECTED_UTILITY_DEFINED")
        self.assertTrue(outcome.terminated)
        self.assertFalse(outcome.truncated)
        self.assertIsNotNone(outcome.reward)
        expected = contract.PILOT_UTILITY_SPEC.expected_utility(
            p_edge_admission_given_sent=outcome.p_edge_admission_given_sent,
            q_perc=outcome.q_perc,
            latency_proxy_ms=outcome.latency_proxy_ms,
        )
        self.assertAlmostEqual(outcome.reward, expected)
        self.assertAlmostEqual(
            outcome.p_edge_admission_given_sent,
            outcome.p_complete_reassembly_given_sent
            * outcome.p_edge_admission_given_reassembled,
        )
        self.assertEqual(outcome.fixed_latency_stages_ms, contract.FIXED_END_TO_FEEDBACK_STAGES_MS)
        self.assertAlmostEqual(
            outcome.latency_proxy_ms,
            113.0 + outcome.conditional_feature_uplink_p50_ms,
        )
        self.assertAlmostEqual(
            outcome.latency_proxy_p95_ms,
            113.0 + outcome.conditional_feature_uplink_p95_ms,
        )
        self.assertAlmostEqual(
            outcome.latency_proxy_p99_ms,
            113.0 + outcome.conditional_feature_uplink_p99_ms,
        )
        self.assertEqual(
            outcome.modeled_budget_miss,
            outcome.latency_proxy_ms > outcome.deadline_ms,
        )
        self.assertIn("NOT_INFERRED", outcome.timeout_probability_status)
        self.assertEqual(result.audit.executed_mode_id, 11)
        self.assertEqual(result.audit.executed_q_e4, 0)
        self.assertIsNone(self.env._active_context)
        self.assertIsNone(self.env._active_radio)

    def test_unsupported_evidence_is_not_relabeled_as_failure(self) -> None:
        outcome = self.env._unavailable_outcome("TEST_UNSUPPORTED")
        self.assertIsNone(outcome.reward)
        self.assertIsNone(outcome.modeled_budget_miss)
        self.assertIn("NOT_RELABELLED", outcome.service_non_admission_semantics)

    def test_environment_checkpoint_round_trip_preserves_draw_streams(self) -> None:
        checkpoint = self.env.state_dict()
        expected_observation, expected_result = self._episode(mode_id=5, q_e4=5902)
        self.env.load_state_dict(checkpoint)
        actual_observation, actual_result = self._episode(mode_id=5, q_e4=5902)
        self.assertEqual(expected_observation, actual_observation)
        self.assertEqual(expected_result, actual_result)

    def test_checkpoint_rejection_is_atomic_and_global_rng_is_untouched(self) -> None:
        checkpoint = self.env.state_dict()
        malformed_radio = replace(
            checkpoint.radio_sampler_state,
            profile_rng_state=("invalid",),
        )
        malformed = replace(
            checkpoint,
            context_rng_state=random.Random(999).getstate(),
            radio_sampler_state=malformed_radio,
        )
        with self.assertRaises(radio.RadioCalibrationError):
            self.env.load_state_dict(malformed)
        self.assertEqual(self.env.state_dict(), checkpoint)

        global_before = random.getstate()
        self._episode(mode_id=3, q_e4=8280)
        self.assertEqual(random.getstate(), global_before)

    def test_training_surface_exposes_no_held_evaluation_api(self) -> None:
        self.assertFalse(hasattr(self.env, "evaluate_held"))
        self.assertFalse(hasattr(self.env._contexts, "evaluate_held"))
        self.assertEqual(len(self.env._contexts.contexts), 476)
        with self.assertRaises(FrozenInstanceError):
            self.env._contexts.contexts[0].sample_id = "held"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

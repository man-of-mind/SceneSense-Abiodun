"""Focused CPU-only tests for the Run-3 reward and simulator kernel."""

from __future__ import annotations

import inspect
import math
import random
import unittest
from dataclasses import replace

from . import empirical_contextual_run3_reward as run3


class Run3RewardContractTest(unittest.TestCase):
    def test_reward_contract_is_literal_hash_bound(self) -> None:
        self.assertEqual(
            run3.RUN3_REWARD_SPEC.canonical_sha256(),
            run3.RUN3_REWARD_SPEC_SHA256,
        )
        self.assertEqual(
            run3.RUN3_REWARD_SPEC_SHA256,
            "f594b204c4ca9bb47ae94d73be20200881cd7771713b17dbe3a9450209ca9841",
        )
        self.assertEqual(run3.RUN3_REWARD_SPEC.quality_weight, 1.0)
        self.assertEqual(run3.RUN3_REWARD_SPEC.latency_weight, 0.25)
        self.assertEqual(run3.RUN3_REWARD_SPEC.deadline_ms, 200.0)
        self.assertEqual(run3.RUN3_REWARD_SPEC.failure_or_timeout_reward, -1.0)
        self.assertEqual(run3.RUN3_REWARD_SPEC.mode_switch_weight, 0.0)
        self.assertEqual(run3.RUN3_REWARD_SPEC.q_switch_weight, 0.0)

    def test_reward_api_accepts_no_probability(self) -> None:
        parameters = inspect.signature(run3.evaluate_run3_reward).parameters
        self.assertEqual(
            tuple(parameters), ("terminal_outcome", "q_perc", "latency_ms")
        )
        self.assertFalse(any(name.startswith("p_") for name in parameters))

    def test_success_formula_and_inclusive_deadline(self) -> None:
        cases = (
            (0.8, 0.0, 0.8),
            (0.8, 100.0, 0.675),
            (0.8, 200.0, 0.55),
        )
        for quality, latency, expected in cases:
            with self.subTest(latency=latency):
                result = run3.evaluate_run3_reward(
                    run3.Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE,
                    quality,
                    latency,
                )
                self.assertEqual(result.scalar_reward, expected)
                self.assertIs(result.learning_eligible, True)
                self.assertIs(result.deadline_met, True)
                self.assertEqual(result.mode_switch_penalty, 0.0)
                self.assertEqual(result.q_switch_penalty, 0.0)
                result.revalidate()

    def test_success_rejects_latency_beyond_deadline(self) -> None:
        with self.assertRaises(run3.Run3RewardError):
            run3.evaluate_run3_reward(
                run3.Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE,
                0.8,
                200.0000001,
            )

    def test_success_rejects_invalid_quality_latency_and_foreign_outcome(self) -> None:
        for quality, latency in (
            (-0.01, 100.0),
            (1.01, 100.0),
            (float("nan"), 100.0),
            (0.8, -0.01),
            (0.8, float("inf")),
            (True, 100.0),
            (0.8, 100),
        ):
            with self.subTest(quality=quality, latency=latency):
                with self.assertRaises(run3.Run3RewardError):
                    run3.evaluate_run3_reward(
                        run3.Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE,
                        quality,
                        latency,
                    )
        with self.assertRaises(run3.Run3RewardError):
            run3.evaluate_run3_reward("SUCCESS_WITHIN_DEADLINE", 0.8, 100.0)

    def test_each_registered_failure_is_exactly_minus_one(self) -> None:
        for terminal in (
            run3.Run3TerminalOutcome.REASSEMBLY_FAILURE,
            run3.Run3TerminalOutcome.EDGE_ADMISSION_FAILURE,
            run3.Run3TerminalOutcome.SIMULATED_SERVICE_TIMEOUT,
        ):
            with self.subTest(terminal=terminal):
                result = run3.evaluate_run3_reward(terminal)
                self.assertEqual(result.scalar_reward, -1.0)
                self.assertIs(result.learning_eligible, True)
                self.assertIs(result.deadline_met, False)
                self.assertIsNone(result.q_perc)
                self.assertIsNone(result.latency_ms)
                result.revalidate()
                with self.assertRaises(run3.Run3RewardError):
                    run3.evaluate_run3_reward(terminal, 0.8, 250.0)

    def test_infrastructure_and_evaluator_faults_are_excluded(self) -> None:
        for terminal in (
            run3.Run3TerminalOutcome.INFRASTRUCTURE_FAULT_EXCLUDED,
            run3.Run3TerminalOutcome.EVALUATOR_FAULT_EXCLUDED,
        ):
            with self.subTest(terminal=terminal):
                result = run3.evaluate_run3_reward(terminal)
                self.assertIsNone(result.scalar_reward)
                self.assertIs(result.learning_eligible, False)
                self.assertIsNone(result.deadline_met)
                self.assertIsNone(result.q_perc)
                self.assertIsNone(result.latency_ms)
                result.revalidate()
                with self.assertRaises(run3.Run3RewardError):
                    run3.evaluate_run3_reward(terminal, 0.8, 100.0)

    def test_reward_result_attestation_detects_post_creation_tamper(self) -> None:
        result = run3.evaluate_run3_reward(
            run3.Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE, 0.8, 100.0
        )
        original = result.scalar_reward
        object.__setattr__(result, "scalar_reward", 0.99)
        with self.assertRaises(run3.Run3RewardError):
            result.revalidate()
        object.__setattr__(result, "scalar_reward", original)
        result.revalidate()

    def test_reward_spec_rejects_unregistered_changes(self) -> None:
        for field, value in (
            ("deadline_ms", 201.0),
            ("latency_weight", 0.2),
            ("mode_switch_weight", 0.01),
            ("q_switch_weight", 0.01),
        ):
            with self.subTest(field=field):
                with self.assertRaises(run3.Run3RewardError):
                    replace(run3.RUN3_REWARD_SPEC, **{field: value})


class Run3KernelContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = run3.QuantileLatencyProxyV1(
            p50_ms=100.0, p95_ms=150.0, p99_ms=200.0
        )

    @staticmethod
    def draws(*, key: str = "decision-1") -> run3.Run3RandomDrawsV1:
        return run3.Run3CounterRngV1(7).draws(key)

    def test_kernel_contract_is_separate_and_literal_hash_bound(self) -> None:
        self.assertEqual(
            run3.RUN3_KERNEL_SPEC.canonical_sha256(),
            run3.RUN3_KERNEL_SPEC_SHA256,
        )
        self.assertEqual(
            run3.RUN3_KERNEL_SPEC_SHA256,
            "1435958ff4df4b0aaf68af02e4113a9b9f3b0c7953b6f73aa5b089a7d2280c02",
        )
        document = run3.RUN3_KERNEL_SPEC.to_canonical_dict()
        self.assertIn("SIMULATOR_TRANSITION_ONLY", document["probability_role"])
        self.assertIn("NOT_RECONSTRUCTED", document["distribution_claim"])
        self.assertIn("NOT_REWARD_INPUT", document["probability_role"])
        self.assertNotEqual(
            run3.RUN3_KERNEL_SPEC_SHA256, run3.RUN3_REWARD_SPEC_SHA256
        )

    def test_quantile_proxy_hits_registered_knots(self) -> None:
        self.assertEqual(self.proxy.lower_endpoint_ms, 50.0)
        expected = {
            0.0: 50.0,
            0.5: 100.0,
            0.95: 150.0,
            0.99: 200.0,
            1.0: 200.0,
        }
        for quantile, latency in expected.items():
            with self.subTest(quantile=quantile):
                self.assertAlmostEqual(self.proxy.inverse_cdf(quantile), latency)
        self.assertAlmostEqual(self.proxy.inverse_cdf(0.725), 125.0)
        self.assertAlmostEqual(self.proxy.inverse_cdf(0.97), 175.0)

    def test_quantile_proxy_is_monotone_and_disclosed_as_proxy(self) -> None:
        values = [self.proxy.inverse_cdf(index / 1000.0) for index in range(1001)]
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        document = self.proxy.to_canonical_dict()
        self.assertIn("NOT_RECONSTRUCTED", document["distribution_claim"])
        self.assertIn("NOT_A_MEASURED", run3.RUN3_EVIDENCE_CLASS)

    def test_quantile_proxy_rejects_invalid_percentiles_and_quantiles(self) -> None:
        for values in (
            (-1.0, 10.0, 20.0),
            (20.0, 10.0, 30.0),
            (10.0, 30.0, 20.0),
            (10.0, 20.0, float("nan")),
        ):
            with self.subTest(values=values):
                with self.assertRaises(run3.Run3RewardError):
                    run3.QuantileLatencyProxyV1(*values)
        for quantile in (-0.01, 1.01, float("nan"), 1):
            with self.subTest(quantile=quantile):
                with self.assertRaises(run3.Run3RewardError):
                    self.proxy.inverse_cdf(quantile)

    def test_counter_rng_is_reproducible_domain_separated_and_local(self) -> None:
        global_state = random.getstate()
        first = run3.Run3CounterRngV1(123).draws("session-A/decision-9")
        second = run3.Run3CounterRngV1(123).draws("session-A/decision-9")
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), global_state)
        values = (first.reassembly_u, first.admission_u, first.latency_u)
        self.assertEqual(len(set(values)), 3)
        self.assertTrue(all(0.0 <= value < 1.0 for value in values))
        self.assertNotEqual(
            first, run3.Run3CounterRngV1(123).draws("session-A/decision-10")
        )
        self.assertNotEqual(
            first, run3.Run3CounterRngV1(124).draws("session-A/decision-9")
        )

    def test_rng_and_draw_records_reject_invalid_inputs(self) -> None:
        for seed in (-1, 1.0, True):
            with self.subTest(seed=seed):
                with self.assertRaises(run3.Run3RewardError):
                    run3.Run3CounterRngV1(seed)
        with self.assertRaises(run3.Run3RewardError):
            run3.Run3CounterRngV1(1).draws("")
        with self.assertRaises(run3.Run3RewardError):
            run3.Run3RandomDrawsV1(
                master_seed=7,
                decision_key="decision-1",
                reassembly_u=1.0,
                admission_u=0.0,
                latency_u=0.0,
            )

    def test_draw_record_rejects_forgery_and_post_creation_tamper(self) -> None:
        authentic = self.draws(key="attested-decision")
        with self.assertRaises(run3.Run3RewardError):
            run3.Run3RandomDrawsV1(
                master_seed=authentic.master_seed,
                decision_key=authentic.decision_key,
                reassembly_u=0.0,
                admission_u=0.0,
                latency_u=0.0,
            )
        original = authentic.reassembly_u
        object.__setattr__(authentic, "reassembly_u", 0.0)
        with self.assertRaises(run3.Run3RewardError):
            authentic.__post_init__()
        object.__setattr__(authentic, "reassembly_u", original)
        authentic.__post_init__()

    def test_reassembly_failure_branch(self) -> None:
        outcome = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=0.0,
            p_edge_admission_given_reassembled=1.0,
            latency_proxy=self.proxy,
            random_draws=self.draws(key="reassembly-failure"),
        )
        self.assertIs(
            outcome.terminal_outcome,
            run3.Run3TerminalOutcome.REASSEMBLY_FAILURE,
        )
        self.assertIs(outcome.reassembled, False)
        self.assertIs(outcome.edge_admitted, False)
        self.assertIsNone(outcome.sampled_latency_ms)
        self.assertEqual(outcome.reward_result.scalar_reward, -1.0)
        outcome.revalidate()

    def test_edge_admission_failure_branch(self) -> None:
        outcome = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=1.0,
            p_edge_admission_given_reassembled=0.0,
            latency_proxy=self.proxy,
            random_draws=self.draws(key="admission-failure"),
        )
        self.assertIs(
            outcome.terminal_outcome,
            run3.Run3TerminalOutcome.EDGE_ADMISSION_FAILURE,
        )
        self.assertIs(outcome.reassembled, True)
        self.assertIs(outcome.edge_admitted, False)
        self.assertIsNone(outcome.sampled_latency_ms)
        self.assertEqual(outcome.reward_result.scalar_reward, -1.0)
        outcome.revalidate()

    def test_timely_success_branch_and_exact_deadline(self) -> None:
        proxy = run3.QuantileLatencyProxyV1(200.0, 200.0, 200.0)
        outcome = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=1.0,
            p_edge_admission_given_reassembled=1.0,
            latency_proxy=proxy,
            random_draws=self.draws(key="exact-deadline"),
        )
        self.assertIs(
            outcome.terminal_outcome,
            run3.Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE,
        )
        self.assertEqual(outcome.sampled_latency_ms, 200.0)
        self.assertEqual(outcome.reward_result.scalar_reward, 0.55)
        outcome.revalidate()

    def test_simulated_timeout_branch_preserves_raw_latency_outside_reward(self) -> None:
        proxy = run3.QuantileLatencyProxyV1(250.0, 250.0, 250.0)
        outcome = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=1.0,
            p_edge_admission_given_reassembled=1.0,
            latency_proxy=proxy,
            random_draws=self.draws(key="simulated-timeout"),
        )
        self.assertIs(
            outcome.terminal_outcome,
            run3.Run3TerminalOutcome.SIMULATED_SERVICE_TIMEOUT,
        )
        self.assertEqual(outcome.sampled_latency_ms, 250.0)
        self.assertIsNone(outcome.reward_result.latency_ms)
        self.assertEqual(outcome.reward_result.scalar_reward, -1.0)
        outcome.revalidate()

    def test_probability_boundaries_use_strict_draw_less_than_probability(self) -> None:
        reassembly_draws = self.draws(key="strict-reassembly-boundary")
        reassembly_zero = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=reassembly_draws.reassembly_u,
            p_edge_admission_given_reassembled=1.0,
            latency_proxy=self.proxy,
            random_draws=reassembly_draws,
        )
        self.assertIs(
            reassembly_zero.terminal_outcome,
            run3.Run3TerminalOutcome.REASSEMBLY_FAILURE,
        )
        admission_draws = self.draws(key="strict-admission-boundary")
        admission_zero = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=1.0,
            p_edge_admission_given_reassembled=admission_draws.admission_u,
            latency_proxy=self.proxy,
            random_draws=admission_draws,
        )
        self.assertIs(
            admission_zero.terminal_outcome,
            run3.Run3TerminalOutcome.EDGE_ADMISSION_FAILURE,
        )

    def test_audit_record_retains_kernel_inputs_but_reward_does_not(self) -> None:
        draws = run3.Run3CounterRngV1(11).draws("audit-decision")
        outcome = run3.sample_run3_simulator_outcome(
            q_perc=0.72,
            p_complete_reassembly_given_sent=0.84,
            p_edge_admission_given_reassembled=0.73,
            latency_proxy=self.proxy,
            random_draws=draws,
        )
        document = outcome.to_canonical_dict()
        self.assertEqual(document["source_q_perc"], 0.72)
        self.assertEqual(document["p_complete_reassembly_given_sent"], 0.84)
        self.assertEqual(document["p_edge_admission_given_reassembled"], 0.73)
        self.assertEqual(document["random_draws"], draws.to_canonical_dict())
        self.assertEqual(document["evidence_class"], run3.RUN3_EVIDENCE_CLASS)
        self.assertEqual(
            document["reward_spec_sha256"], run3.RUN3_REWARD_SPEC_SHA256
        )
        self.assertEqual(
            document["kernel_spec_sha256"], run3.RUN3_KERNEL_SPEC_SHA256
        )
        reward_document = document["reward_result"]
        self.assertFalse(any(key.startswith("p_") for key in reward_document))

    def test_simulated_outcome_attestation_detects_tamper(self) -> None:
        outcome = run3.sample_run3_simulator_outcome(
            q_perc=0.8,
            p_complete_reassembly_given_sent=1.0,
            p_edge_admission_given_reassembled=1.0,
            latency_proxy=self.proxy,
            random_draws=self.draws(key="tamper-outcome"),
        )
        original = outcome.sampled_latency_ms
        object.__setattr__(outcome, "sampled_latency_ms", original + 0.1)
        with self.assertRaises(run3.Run3RewardError):
            outcome.revalidate()
        object.__setattr__(outcome, "sampled_latency_ms", original)
        outcome.revalidate()


class Run3ExpectedAnalysisTest(unittest.TestCase):
    def test_constant_timely_latency_has_closed_form_expectation(self) -> None:
        result = run3.expected_run3_reward(
            q_perc=0.8,
            p_complete_reassembly_given_sent=0.5,
            p_edge_admission_given_reassembled=0.5,
            latency_proxy=run3.QuantileLatencyProxyV1(100.0, 100.0, 100.0),
        )
        success_reward = 0.8 - 0.25 * 100.0 / 200.0
        self.assertAlmostEqual(result.expected_reward, 0.25 * success_reward - 0.75)
        self.assertEqual(result.p_reassembly_failure, 0.5)
        self.assertEqual(result.p_admission_failure, 0.25)
        self.assertEqual(result.p_admitted, 0.25)
        self.assertEqual(result.p_timeout_given_admitted, 0.0)
        self.assertEqual(result.p_timely_feedback, 0.25)

    def test_constant_late_latency_is_always_minus_one(self) -> None:
        result = run3.expected_run3_reward(
            q_perc=0.8,
            p_complete_reassembly_given_sent=0.9,
            p_edge_admission_given_reassembled=0.8,
            latency_proxy=run3.QuantileLatencyProxyV1(250.0, 250.0, 250.0),
        )
        self.assertAlmostEqual(result.expected_reward, -1.0)
        self.assertEqual(result.p_timeout_given_admitted, 1.0)
        self.assertEqual(result.p_timely_feedback, 0.0)

    def test_expected_analysis_matches_deterministic_monte_carlo(self) -> None:
        quality = 0.7
        p_reassembly = 0.8
        p_admission = 0.75
        proxy = run3.QuantileLatencyProxyV1(150.0, 220.0, 260.0)
        analytic = run3.expected_run3_reward(
            q_perc=quality,
            p_complete_reassembly_given_sent=p_reassembly,
            p_edge_admission_given_reassembled=p_admission,
            latency_proxy=proxy,
        )
        rng = run3.Run3CounterRngV1(90210)
        observed = 0.0
        count = 2500
        for index in range(count):
            outcome = run3.sample_run3_simulator_outcome(
                q_perc=quality,
                p_complete_reassembly_given_sent=p_reassembly,
                p_edge_admission_given_reassembled=p_admission,
                latency_proxy=proxy,
                random_draws=rng.draws(f"monte-carlo/{index}"),
            )
            observed += outcome.reward_result.scalar_reward
        observed /= count
        self.assertAlmostEqual(observed, analytic.expected_reward, delta=0.04)

    def test_expected_value_is_analysis_only_not_reward_api(self) -> None:
        signature = inspect.signature(run3.evaluate_run3_reward)
        self.assertNotIn("p_complete_reassembly_given_sent", signature.parameters)
        self.assertNotIn("p_edge_admission_given_reassembled", signature.parameters)
        self.assertIn(
            "p_complete_reassembly_given_sent",
            inspect.signature(run3.expected_run3_reward).parameters,
        )

    def test_module_has_no_torch_dependency(self) -> None:
        self.assertNotIn("torch", run3.__dict__)


if __name__ == "__main__":
    unittest.main()

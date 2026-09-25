"""Adversarial tests for the prospective Run-4 smoke preregistration."""

from __future__ import annotations

import builtins
import importlib
import random
import socket
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

from . import smoke_preregistration as subject


CONTEXT_COUNT = 100


def _panel() -> subject.DiagnosticPanelManifestV1:
    return subject.DiagnosticPanelManifestV1(
        validation_calibration_cell_ids=("cal-v-0", "cal-v-1"),
        validation_calibration_cells_sha256="a" * 64,
        fit_validation_scene_ids=("scene-v-0", "scene-v-1", "scene-v-2"),
        fit_validation_scenes_sha256="b" * 64,
        ordered_context_ids=tuple(
            f"context-{index:03d}" for index in range(CONTEXT_COUNT)
        ),
        ordered_context_rows_sha256="c" * 64,
        ordered_context_group_ids=tuple(
            f"scene-group-{index // 4:03d}" for index in range(CONTEXT_COUNT)
        ),
        ordered_context_groups_sha256="5" * 64,
        queue_kernel_binding_sha256="d" * 64,
        widest_support_mode_ids=(11,),
        fixed_comparator_mode_id=8,
        fixed_comparator_q_e4=7000,
        fixed_comparator_fit_selection_sha256="6" * 64,
        exact_oracle_evaluator_sha256="7" * 64,
        predeclared_runtime_limit_seconds=100.0,
        runtime_budget_registration_sha256="e" * 64,
    )


def _sensitivity(
    feature: str,
    *,
    effect: float = 0.2,
    lower: float = 0.1,
    upper: float = 0.3,
    changed_only: bool = True,
) -> subject.SensitivityDiagnosticV1:
    return subject.SensitivityDiagnosticV1(
        feature_name=feature,
        controlled_pair_count=20,
        nonzero_response_count=12,
        effect_estimate=effect,
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
        confidence_level=0.95,
        only_registered_feature_changed=changed_only,
        report_sha256=feature.encode("ascii").hex()[:1].ljust(64, "f"),
    )


def _checkpoint(
    update: int,
    *,
    policy_reward: float | None = None,
    counts: tuple[int, ...] | None = None,
) -> subject.CheckpointDiagnosticV1:
    values = {
        0: (0.30, 0.40, 0.50, 220.0, 300.0, 0.10, 0.40),
        100: (0.36, 0.30, 0.52, 210.0, 280.0, 0.20, 0.50),
        250: (0.42, 0.20, 0.54, 190.0, 250.0, 0.30, 0.60),
        500: (0.50, 0.10, 0.56, 160.0, 220.0, 0.40, 0.70),
    }
    expected, misses, qperc, p50, p95, rank, q_mean = values[update]
    if policy_reward is not None:
        expected = policy_reward
    oracle = 0.60
    total = oracle - expected
    continuous = total * 0.4
    discrete = total - continuous
    if counts is None:
        counts_list = [0] * 12
        counts_list[8] = 60
        counts_list[11] = 40
        counts = tuple(counts_list)
    return subject.CheckpointDiagnosticV1(
        update=update,
        panel_context_count=CONTEXT_COUNT,
        expected_policy_reward=expected,
        fixed_comparator_reward=0.20,
        oracle_reward=oracle,
        oracle_gap=total,
        deadline_miss_rate=misses,
        fixed_comparator_deadline_miss_rate=0.25,
        mean_action_qperc=qperc,
        successful_feedback_latency_p50_ms=p50,
        successful_feedback_latency_p95_ms=p95,
        critic_rank_correlation=rank,
        deterministic_mode_counts=counts,
        q_mean=q_mean,
        q_std=0.10,
        q_support_boundary_hit_rate=0.05,
        total_regret=total,
        discrete_mode_regret=discrete,
        continuous_q_regret=continuous,
        context_rows_sha256="c" * 64,
    )


def _diagnostics(
    panel: subject.DiagnosticPanelManifestV1,
) -> subject.SmokeDiagnosticsV1:
    return subject.SmokeDiagnosticsV1(
        preregistration_sha256=subject.PREREGISTRATION_SHA256,
        panel_sha256=panel.canonical_sha256,
        seed=17,
        checkpoint_diagnostics=tuple(
            _checkpoint(update) for update in (0, 100, 250, 500)
        ),
        observed_mode_q_bins=tuple(
            (mode_id, q_bin)
            for mode_id in range(12)
            for q_bin in range(6)
        ),
        registered_success_count=50,
        registered_failure_count=4,
        all_numerics_finite=True,
        policy_minus_fixed_ci_lower=0.20,
        policy_minus_fixed_ci_upper=0.40,
        oracle_gap_reduction_ci_lower=0.15,
        oracle_gap_reduction_ci_upper=0.25,
        paired_confidence_level=0.95,
        paired_interval_spec_sha256=subject.PAIRED_INTERVAL_SPEC_SHA256,
        sensitivity_diagnostics=tuple(
            _sensitivity(feature) for feature in subject.SENSITIVITY_FEATURES
        ),
        checkpoint_resume_from_update=250,
        checkpoint_resume_bit_identical=True,
        runtime_seconds=90.0,
        checkpoint_manifest_sha256="1" * 64,
        diagnostic_report_sha256="2" * 64,
    )


def _evidence(
    panel: subject.DiagnosticPanelManifestV1,
) -> subject.CompositeVerifierEvidenceV1:
    return subject.CompositeVerifierEvidenceV1(
        panel=panel,
        composite_verifier_manifest_sha256="9" * 64,
        calibration_cells_disjoint_from_fit=True,
        scenes_disjoint_from_fit=True,
        ordered_contexts_deterministic=True,
        observed_row_count=len(panel.ordered_context_ids),
        fabricated_row_count=0,
        kernel_binding_verified=True,
        identities_and_digests_verified=True,
        fixed_comparator_fit_only_verified=True,
        exact_oracle_verified=True,
        paired_interval_spec_verified=True,
    )


class FrozenConfigurationTest(unittest.TestCase):
    def test_requested_configuration_is_frozen_exactly(self) -> None:
        config = subject.FROZEN_CONFIG
        self.assertEqual(config.gamma_per_tensor, 0.99)
        self.assertEqual(config.alpha_d, 0.05)
        self.assertEqual(config.alpha_c, 0.02)
        self.assertEqual(config.actor_learning_rate, 3e-4)
        self.assertEqual(config.critic_learning_rate, 3e-4)
        self.assertEqual(config.polyak_tau, 0.005)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.replay_capacity, 65536)
        self.assertEqual(config.environment_transitions_per_update, 4)
        self.assertEqual(config.torch_intraop_threads, 4)
        self.assertEqual(config.seed_order, (17, 29, 43))
        self.assertEqual(config.checkpoint_updates, (0, 100, 250, 500, 1500, 10000))
        self.assertEqual(config.paired_confidence_level, 0.95)
        self.assertEqual(config.paired_bootstrap_resamples, 10_000)
        self.assertEqual(config.paired_bootstrap_seed, 17)
        self.assertEqual(config.max_dominant_mode_fraction, 0.95)

    def test_warmup_is_exactly_12_by_6_by_4(self) -> None:
        config = subject.FROZEN_CONFIG
        self.assertEqual(config.warmup_mode_count, 12)
        self.assertEqual(config.warmup_q_bin_count, 6)
        self.assertEqual(config.warmup_samples_per_mode_q_bin, 4)
        self.assertEqual(config.warmup_decision_count, 288)

    def test_seed_17_alone_precedes_continuation(self) -> None:
        self.assertEqual(subject.FROZEN_CONFIG.initial_smoke_seed, 17)
        self.assertEqual(subject.FROZEN_CONFIG.smoke_stop_update, 500)
        self.assertEqual(subject.FROZEN_CONFIG.smoke_checkpoint_updates, (0, 100, 250, 500))

    def test_discount_rule_says_replay_not_trainer_recomputation(self) -> None:
        text = subject.FROZEN_CONFIG.to_canonical_dict()["discount_rule"]
        self.assertIn("replay", text)
        self.assertIn("never recomputes", text)

    def test_config_rejects_drift(self) -> None:
        with self.assertRaisesRegex(Exception, "mode x q-bin"):
            replace(subject.FROZEN_CONFIG, warmup_decision_count=287)
        with self.assertRaisesRegex(Exception, "first seed"):
            replace(subject.FROZEN_CONFIG, seed_order=(29, 17, 43))
        with self.assertRaisesRegex(Exception, "0/100/250/500/1500/10000"):
            replace(subject.FROZEN_CONFIG, checkpoint_updates=(0, 100, 250, 500, 1000, 10000))
        with self.assertRaisesRegex(Exception, "\(0,1\)"):
            replace(subject.FROZEN_CONFIG, max_dominant_mode_fraction=1.0)

    def test_preregistration_digest_is_deterministic(self) -> None:
        self.assertEqual(len(subject.PREREGISTRATION_SHA256), 64)
        self.assertEqual(
            subject.PREREGISTRATION_SHA256,
            subject._canonical_sha256(
                {
                    "config": subject.FROZEN_CONFIG.to_canonical_dict(),
                    "gates": [item.to_canonical_dict() for item in subject.FROZEN_GATE_SPECS],
                    "paired_interval_spec_sha256": subject.PAIRED_INTERVAL_SPEC_SHA256,
                    "record_type": "run4_smoke_preregistration_bundle_v1",
                    "schema_id": subject.SCHEMA_ID,
                    "schema_version": subject.SCHEMA_VERSION,
                }
            ),
        )

    def test_gate_set_uses_real_comparators_not_reward_near_one(self) -> None:
        gate_ids = {item.gate_id for item in subject.FROZEN_GATE_SPECS}
        self.assertIn("policy_reward_beats_fixed_comparator", gate_ids)
        self.assertIn("oracle_gap_reduction", gate_ids)
        self.assertIn("no_near_total_mode_collapse", gate_ids)
        self.assertNotIn("controlled_state_sensitivity", gate_ids)
        self.assertFalse(any("reward near 1" in item.rule.lower() for item in subject.FROZEN_GATE_SPECS))


class PanelManifestTest(unittest.TestCase):
    def test_panel_binds_fit_selected_comparator_and_exact_oracle(self) -> None:
        payload = _panel().to_canonical_dict()
        self.assertEqual(payload["fixed_comparator_fit_selection_partition"], "FIT_ONLY")
        self.assertEqual(payload["fixed_comparator_mode_id"], 8)
        self.assertEqual(payload["fixed_comparator_q_e4"], 7000)
        self.assertEqual(payload["validation_calibration_partition"], "VALIDATION_ONLY")

    def test_empty_duplicate_or_foreign_identity_is_refused(self) -> None:
        with self.assertRaises(Exception):
            replace(_panel(), validation_calibration_cell_ids=())
        with self.assertRaisesRegex(Exception, "duplicate"):
            replace(_panel(), fit_validation_scene_ids=("same", "same"))
        with self.assertRaisesRegex(Exception, "registered mode"):
            replace(_panel(), fixed_comparator_mode_id=12)
        with self.assertRaisesRegex(Exception, "wire range"):
            replace(_panel(), fixed_comparator_q_e4=9801)
        with self.assertRaisesRegex(Exception, "64 lowercase"):
            replace(_panel(), exact_oracle_evaluator_sha256="unknown")
        with self.assertRaisesRegex(Exception, "one-for-one"):
            replace(_panel(), ordered_context_group_ids=("one", "two"))
        with self.assertRaisesRegex(Exception, "at least two"):
            replace(
                _panel(),
                ordered_context_group_ids=("same",) * CONTEXT_COUNT,
            )

    def test_runtime_budget_and_widest_modes_are_validated(self) -> None:
        with self.assertRaisesRegex(Exception, "positive"):
            replace(_panel(), predeclared_runtime_limit_seconds=0.0)
        with self.assertRaisesRegex(Exception, "outside"):
            replace(_panel(), widest_support_mode_ids=(12,))
        with self.assertRaisesRegex(Exception, "duplicates"):
            replace(_panel(), widest_support_mode_ids=(11, 11))


class ProductionEvidenceGateTest(unittest.TestCase):
    def test_current_source_is_deliberately_fail_closed(self) -> None:
        self.assertIsNone(subject.REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256)
        with self.assertRaisesRegex(Exception, "no composite verifier"):
            subject.bind_verified_panel(_evidence(_panel()))

    def test_false_disjointness_fabrication_and_test_evidence_are_refused(self) -> None:
        evidence = _evidence(_panel())
        with self.assertRaisesRegex(Exception, "overlap"):
            subject.bind_verified_panel(replace(evidence, scenes_disjoint_from_fit=False))
        with self.assertRaisesRegex(Exception, "fabricated"):
            subject.bind_verified_panel(replace(evidence, fabricated_row_count=1))
        test_evidence = replace(evidence, evidence_class=subject.EvidenceClass.TEST_ONLY_SYNTHETIC)
        with self.assertRaisesRegex(Exception, "test-only"):
            subject.bind_verified_panel(test_evidence)
        with self.assertRaisesRegex(Exception, "fit only"):
            subject.bind_verified_panel(
                replace(evidence, fixed_comparator_fit_only_verified=False)
            )
        with self.assertRaisesRegex(Exception, "oracle"):
            subject.bind_verified_panel(replace(evidence, exact_oracle_verified=False))
        with self.assertRaisesRegex(Exception, "interval"):
            subject.bind_verified_panel(
                replace(evidence, paired_interval_spec_verified=False)
            )

    def test_registered_manifest_can_issue_exact_private_binding(self) -> None:
        evidence = _evidence(_panel())
        with mock.patch.object(
            subject,
            "REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256",
            evidence.composite_verifier_manifest_sha256,
        ):
            binding = subject.bind_verified_panel(evidence)
        binding.require_verified()

    def test_forged_private_binding_token_is_refused(self) -> None:
        binding = subject._VerifiedPanelBindingV1(
            panel=_panel(),
            verifier_evidence_sha256="3" * 64,
            preregistration_sha256=subject.PREREGISTRATION_SHA256,
            evidence_class=subject.EvidenceClass.VERIFIED_COMPOSITE,
            _token=object(),
        )
        with self.assertRaisesRegex(Exception, "not verifier-issued"):
            binding.require_verified()


class CheckpointDefinitionTest(unittest.TestCase):
    def test_all_required_metrics_are_canonicalized(self) -> None:
        record = _checkpoint(500)
        payload = record.to_canonical_dict()
        for name in (
            "expected_policy_reward",
            "fixed_comparator_reward",
            "oracle_reward",
            "oracle_gap",
            "deadline_miss_rate",
            "mean_action_qperc",
            "successful_feedback_latency_p50_ms",
            "successful_feedback_latency_p95_ms",
            "mode_entropy_nats",
            "dominant_mode_fraction",
            "q_mean",
            "q_std",
            "q_support_boundary_hit_rate",
            "total_regret",
            "discrete_mode_regret",
            "continuous_q_regret",
        ):
            self.assertIn(name, payload)

    def test_entropy_and_dominance_are_derived_from_counts(self) -> None:
        record = _checkpoint(500)
        expected = -(0.6 * __import__("math").log(0.6) + 0.4 * __import__("math").log(0.4))
        self.assertAlmostEqual(record.mode_entropy_nats, expected)
        self.assertEqual(record.dominant_mode_fraction, 0.6)

    def test_reward_oracle_and_regret_definitions_fail_closed(self) -> None:
        record = _checkpoint(500)
        with self.assertRaisesRegex(Exception, "oracle_gap contradicts"):
            replace(record, oracle_gap=0.2)
        with self.assertRaisesRegex(Exception, "total_regret"):
            replace(record, total_regret=0.2)
        with self.assertRaisesRegex(Exception, "add to total"):
            replace(record, discrete_mode_regret=0.01)
        with self.assertRaisesRegex(Exception, "cannot exceed"):
            replace(record, expected_policy_reward=0.7)

    def test_domains_and_percentile_order_fail_closed(self) -> None:
        record = _checkpoint(500)
        with self.assertRaisesRegex(Exception, "P50 <= P95"):
            replace(record, successful_feedback_latency_p95_ms=100.0)
        with self.assertRaisesRegex(Exception, "\[0,1\]"):
            replace(record, deadline_miss_rate=1.1)
        with self.assertRaisesRegex(Exception, "executable q range"):
            replace(record, q_mean=0.981)
        with self.assertRaisesRegex(Exception, "cover the panel"):
            replace(record, deterministic_mode_counts=(0,) * 12)


class DiagnosticShapeTest(unittest.TestCase):
    def test_passing_fixture_has_all_four_checkpoints(self) -> None:
        diagnostics = _diagnostics(_panel())
        self.assertEqual(diagnostics.evaluated_checkpoint_updates, (0, 100, 250, 500))
        self.assertEqual(diagnostics.checkpoint(500).expected_policy_reward, 0.5)
        self.assertEqual(len(diagnostics.canonical_sha256), 64)

    def test_checkpoint_set_and_panel_rows_must_match_exactly(self) -> None:
        diagnostics = _diagnostics(_panel())
        with self.assertRaisesRegex(Exception, "0/100/250/500"):
            replace(diagnostics, checkpoint_diagnostics=diagnostics.checkpoint_diagnostics[:-1])
        changed = replace(diagnostics.checkpoint_diagnostics[1], context_rows_sha256="8" * 64)
        with self.assertRaisesRegex(Exception, "same exact panel"):
            replace(
                diagnostics,
                checkpoint_diagnostics=(diagnostics.checkpoint_diagnostics[0], changed) + diagnostics.checkpoint_diagnostics[2:],
            )

    def test_fixed_and_oracle_values_are_checkpoint_invariant(self) -> None:
        diagnostics = _diagnostics(_panel())
        changed = replace(diagnostics.checkpoint_diagnostics[2], fixed_comparator_reward=0.21)
        with self.assertRaisesRegex(Exception, "checkpoint-invariant"):
            replace(
                diagnostics,
                checkpoint_diagnostics=diagnostics.checkpoint_diagnostics[:2] + (changed,) + diagnostics.checkpoint_diagnostics[3:],
            )

    def test_confidence_intervals_must_contain_their_paired_estimates(self) -> None:
        diagnostics = _diagnostics(_panel())
        with self.assertRaisesRegex(Exception, "outside"):
            replace(diagnostics, policy_minus_fixed_ci_lower=0.31)
        with self.assertRaisesRegex(Exception, "outside"):
            replace(diagnostics, oracle_gap_reduction_ci_upper=0.19)
        with self.assertRaisesRegex(Exception, "differs"):
            replace(diagnostics, paired_confidence_level=0.90)
        with self.assertRaisesRegex(Exception, "method differs"):
            replace(diagnostics, paired_interval_spec_sha256="8" * 64)

    def test_sensitivity_records_remain_complete(self) -> None:
        diagnostics = _diagnostics(_panel())
        duplicate = diagnostics.sensitivity_diagnostics[:-1] + (diagnostics.sensitivity_diagnostics[0],)
        with self.assertRaisesRegex(Exception, "every registered feature"):
            replace(diagnostics, sensitivity_diagnostics=duplicate)


class GateMechanicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = _panel()
        self.binding = subject._TestOnlyPanelBindingV1(self.panel)
        self.base = _diagnostics(self.panel)

    def _assessment(self, diagnostics=None):
        return subject._assess_test_only_smoke(
            self.binding, self.base if diagnostics is None else diagnostics
        )

    def _failed(self, diagnostics) -> set[str]:
        return {
            item.gate_id
            for item in self._assessment(diagnostics).gate_results
            if not item.passed
        }

    def _replace_checkpoint(
        self,
        update: int,
        replacement: subject.CheckpointDiagnosticV1,
        **diagnostic_changes,
    ):
        return replace(
            self.base,
            checkpoint_diagnostics=tuple(
                replacement if item.update == update else item
                for item in self.base.checkpoint_diagnostics
            ),
            **diagnostic_changes,
        )

    def test_all_pass_test_fixture_still_cannot_authorize(self) -> None:
        assessment = self._assessment()
        self.assertTrue(assessment.all_gates_pass)
        self.assertFalse(assessment.continuation_authorized)
        with self.assertRaises(subject.ContinuationRefused):
            assessment.require_continuation_authorized()

    def test_integrity_gates(self) -> None:
        observed = tuple(item for item in self.base.observed_mode_q_bins if item != (11, 5))
        self.assertIn("warmup_mode_q_coverage", self._failed(replace(self.base, observed_mode_q_bins=observed)))
        self.assertIn("feedback_outcome_coverage", self._failed(replace(self.base, registered_failure_count=0)))
        self.assertIn("finite_numerics", self._failed(replace(self.base, all_numerics_finite=False)))
        self.assertIn("checkpoint_resume_bit_identity", self._failed(replace(self.base, checkpoint_resume_bit_identical=False)))
        self.assertIn("runtime_within_predeclared_bound", self._failed(replace(self.base, runtime_seconds=100.0001)))

    def test_policy_must_beat_fixed_with_ci_excluding_zero(self) -> None:
        diagnostics = replace(
            self.base,
            policy_minus_fixed_ci_lower=-0.01,
            policy_minus_fixed_ci_upper=0.40,
        )
        self.assertIn("policy_reward_beats_fixed_comparator", self._failed(diagnostics))

    def test_epsilon_reward_improvement_does_not_pass_when_ci_crosses_zero(self) -> None:
        update500 = _checkpoint(500, policy_reward=0.200000001)
        diagnostics = self._replace_checkpoint(
            500,
            update500,
            policy_minus_fixed_ci_lower=-1e-6,
            policy_minus_fixed_ci_upper=1e-6,
            oracle_gap_reduction_ci_lower=-0.11,
            oracle_gap_reduction_ci_upper=-0.09,
        )
        self.assertIn("policy_reward_beats_fixed_comparator", self._failed(diagnostics))

    def test_oracle_gap_reduction_needs_paired_ci_excluding_zero(self) -> None:
        diagnostics = replace(
            self.base,
            oracle_gap_reduction_ci_lower=-0.01,
            oracle_gap_reduction_ci_upper=0.25,
        )
        self.assertIn("oracle_gap_reduction", self._failed(diagnostics))

    def test_99_percent_mode_collapse_fails_and_95_percent_is_boundary(self) -> None:
        counts99 = (99, 1) + (0,) * 10
        collapsed = replace(self.base.checkpoint(500), deterministic_mode_counts=counts99)
        diagnostics = self._replace_checkpoint(500, collapsed)
        self.assertIn("no_near_total_mode_collapse", self._failed(diagnostics))
        counts95 = (95, 5) + (0,) * 10
        boundary = replace(self.base.checkpoint(500), deterministic_mode_counts=counts95)
        diagnostics = self._replace_checkpoint(500, boundary)
        self.assertNotIn("no_near_total_mode_collapse", self._failed(diagnostics))

    def test_sensitivity_is_diagnostic_only(self) -> None:
        original = self.base.sensitivity_diagnostics[0]
        inconclusive = replace(
            original,
            effect_estimate=0.0,
            confidence_interval_lower=-0.1,
            confidence_interval_upper=0.1,
            nonzero_response_count=0,
            only_registered_feature_changed=False,
        )
        diagnostics = replace(
            self.base,
            sensitivity_diagnostics=(inconclusive,) + self.base.sensitivity_diagnostics[1:],
        )
        self.assertTrue(self._assessment(diagnostics).all_gates_pass)

    def test_component_regrets_are_recorded_but_not_independent_gates(self) -> None:
        update500 = self.base.checkpoint(500)
        shifted = replace(
            update500,
            discrete_mode_regret=0.01,
            continuous_q_regret=0.09,
        )
        diagnostics = self._replace_checkpoint(500, shifted)
        self.assertTrue(self._assessment(diagnostics).all_gates_pass)

    def test_wrong_panel_seed_or_preregistration_is_refused_not_scored(self) -> None:
        with self.assertRaisesRegex(Exception, "another diagnostic panel"):
            self._assessment(replace(self.base, panel_sha256="4" * 64))
        with self.assertRaisesRegex(Exception, "seed 17"):
            self._assessment(replace(self.base, seed=29))
        with self.assertRaisesRegex(Exception, "another preregistration"):
            self._assessment(replace(self.base, preregistration_sha256="5" * 64))

    def test_verified_all_pass_path_can_authorize_after_registration(self) -> None:
        evidence = _evidence(self.panel)
        with mock.patch.object(
            subject,
            "REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256",
            evidence.composite_verifier_manifest_sha256,
        ):
            binding = subject.bind_verified_panel(evidence)
        assessment = subject.assess_verified_smoke(binding, self.base)
        self.assertTrue(assessment.continuation_authorized)


class ImportPurityTest(unittest.TestCase):
    def test_import_reads_nothing_and_starts_nothing(self) -> None:
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("filesystem read")
        ) as opened, mock.patch.object(random, "seed") as seeded, mock.patch.object(
            random, "random"
        ) as sampled, mock.patch.object(
            socket, "socket", side_effect=AssertionError("socket opened")
        ) as socket_opened, mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("process started")
        ) as popen:
            importlib.reload(subject)
        opened.assert_not_called()
        seeded.assert_not_called()
        sampled.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()

    def test_test_only_symbols_are_not_public(self) -> None:
        self.assertNotIn("_TestOnlyPanelBindingV1", subject.__all__)
        self.assertNotIn("_assess_test_only_smoke", subject.__all__)
        self.assertNotIn("_VerifiedPanelBindingV1", subject.__all__)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

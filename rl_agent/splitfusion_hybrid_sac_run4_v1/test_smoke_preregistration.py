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


def _panel() -> subject.DiagnosticPanelManifestV1:
    return subject.DiagnosticPanelManifestV1(
        validation_calibration_cell_ids=("cal-v-0", "cal-v-1"),
        validation_calibration_cells_sha256="a" * 64,
        fit_validation_scene_ids=("scene-v-0", "scene-v-1", "scene-v-2"),
        fit_validation_scenes_sha256="b" * 64,
        ordered_context_ids=tuple(f"context-{index}" for index in range(6)),
        ordered_context_rows_sha256="c" * 64,
        queue_kernel_binding_sha256="d" * 64,
        widest_support_mode_ids=(11,),
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


def _diagnostics(
    panel: subject.DiagnosticPanelManifestV1,
) -> subject.SmokeDiagnosticsV1:
    counts = [0] * 12
    counts[0] = 3
    counts[11] = 3
    return subject.SmokeDiagnosticsV1(
        preregistration_sha256=subject.PREREGISTRATION_SHA256,
        panel_sha256=panel.canonical_sha256,
        seed=17,
        evaluated_checkpoint_updates=(0, 100, 250, 500),
        observed_mode_q_bins=tuple(
            (mode_id, q_bin)
            for mode_id in range(12)
            for q_bin in range(6)
        ),
        registered_success_count=50,
        registered_failure_count=4,
        all_numerics_finite=True,
        critic_rank_correlation_update0=0.1,
        critic_rank_correlation_update500=0.2,
        action_regret_update0=0.3,
        action_regret_update500=0.2,
        continuous_q_regret_update0=0.2,
        continuous_q_regret_update500=0.1,
        deterministic_mode_counts=tuple(counts),
        panel_context_count=6,
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

    def test_warmup_is_exactly_12_by_6_by_4(self) -> None:
        config = subject.FROZEN_CONFIG
        self.assertEqual(config.warmup_mode_count, 12)
        self.assertEqual(config.warmup_q_bin_count, 6)
        self.assertEqual(config.warmup_samples_per_mode_q_bin, 4)
        self.assertEqual(config.warmup_decision_count, 288)

    def test_seed_17_alone_precedes_continuation(self) -> None:
        self.assertEqual(subject.FROZEN_CONFIG.initial_smoke_seed, 17)
        self.assertEqual(subject.FROZEN_CONFIG.smoke_stop_update, 500)
        self.assertEqual(
            subject.FROZEN_CONFIG.smoke_checkpoint_updates,
            (0, 100, 250, 500),
        )

    def test_discount_rule_says_replay_not_trainer_recomputation(self) -> None:
        text = subject.FROZEN_CONFIG.to_canonical_dict()["discount_rule"]
        self.assertIn("replay", text)
        self.assertIn("never recomputes", text)

    def test_config_rejects_drifted_warmup_total(self) -> None:
        with self.assertRaisesRegex(Exception, "mode x q-bin"):
            replace(subject.FROZEN_CONFIG, warmup_decision_count=287)

    def test_config_rejects_seed_reordering(self) -> None:
        with self.assertRaisesRegex(Exception, "first seed"):
            replace(subject.FROZEN_CONFIG, seed_order=(29, 17, 43))

    def test_config_rejects_checkpoint_drift(self) -> None:
        with self.assertRaisesRegex(Exception, "0/100/250/500/1500/10000"):
            replace(
                subject.FROZEN_CONFIG,
                checkpoint_updates=(0, 100, 250, 500, 1000, 10000),
            )

    def test_preregistration_digest_is_deterministic(self) -> None:
        self.assertEqual(len(subject.PREREGISTRATION_SHA256), 64)
        self.assertEqual(
            subject.PREREGISTRATION_SHA256,
            subject._canonical_sha256(
                {
                    "config": subject.FROZEN_CONFIG.to_canonical_dict(),
                    "gates": [
                        item.to_canonical_dict()
                        for item in subject.FROZEN_GATE_SPECS
                    ],
                    "record_type": "run4_smoke_preregistration_bundle_v1",
                    "schema_id": subject.SCHEMA_ID,
                    "schema_version": subject.SCHEMA_VERSION,
                }
            ),
        )

    def test_gate_labels_separate_integrity_from_hypotheses(self) -> None:
        classifications = {
            item.gate_id: item.classification
            for item in subject.FROZEN_GATE_SPECS
        }
        self.assertIs(
            classifications["finite_numerics"],
            subject.GateClassification.INTEGRITY_ACCEPTANCE,
        )
        self.assertIs(
            classifications["critic_rank_direction"],
            subject.GateClassification.HYPOTHESIS_DIAGNOSTIC,
        )
        self.assertNotIn("threshold", classifications["action_regret_direction"].value)


class PanelManifestTest(unittest.TestCase):
    def test_panel_is_canonical_and_hashable(self) -> None:
        panel = _panel()
        self.assertEqual(len(panel.canonical_sha256), 64)
        self.assertEqual(panel.canonical_sha256, _panel().canonical_sha256)
        self.assertIsInstance(hash(panel), int)
        payload = panel.to_canonical_dict()
        self.assertEqual(payload["validation_calibration_partition"], "VALIDATION_ONLY")
        self.assertEqual(payload["fit_validation_scene_partition"], "FIT_VALIDATION_ONLY")

    def test_empty_or_duplicate_source_identities_are_refused(self) -> None:
        with self.assertRaises(Exception):
            replace(_panel(), validation_calibration_cell_ids=())
        with self.assertRaisesRegex(Exception, "duplicate"):
            replace(_panel(), fit_validation_scene_ids=("same", "same"))
        with self.assertRaisesRegex(Exception, "duplicate"):
            replace(_panel(), ordered_context_ids=("same", "same"))

    def test_bad_hash_is_refused(self) -> None:
        with self.assertRaisesRegex(Exception, "64 lowercase"):
            replace(_panel(), queue_kernel_binding_sha256="unknown")

    def test_runtime_budget_must_be_positive_and_prebound(self) -> None:
        with self.assertRaisesRegex(Exception, "positive"):
            replace(_panel(), predeclared_runtime_limit_seconds=0.0)
        with self.assertRaisesRegex(Exception, "64 lowercase"):
            replace(_panel(), runtime_budget_registration_sha256="late")

    def test_widest_modes_must_be_unique_registered_modes(self) -> None:
        with self.assertRaisesRegex(Exception, "outside"):
            replace(_panel(), widest_support_mode_ids=(12,))
        with self.assertRaisesRegex(Exception, "duplicates"):
            replace(_panel(), widest_support_mode_ids=(11, 11))


class ProductionEvidenceGateTest(unittest.TestCase):
    def test_current_source_is_deliberately_fail_closed(self) -> None:
        self.assertIsNone(subject.REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256)
        with self.assertRaisesRegex(Exception, "no composite verifier"):
            subject.bind_verified_panel(_evidence(_panel()))

    def test_false_disjointness_and_fabrication_are_refused(self) -> None:
        evidence = _evidence(_panel())
        with self.assertRaisesRegex(Exception, "overlap"):
            subject.bind_verified_panel(
                replace(evidence, scenes_disjoint_from_fit=False)
            )
        with self.assertRaisesRegex(Exception, "fabricated"):
            subject.bind_verified_panel(replace(evidence, fabricated_row_count=1))

    def test_test_evidence_cannot_claim_production(self) -> None:
        evidence = replace(
            _evidence(_panel()),
            evidence_class=subject.EvidenceClass.TEST_ONLY_SYNTHETIC,
        )
        with self.assertRaisesRegex(Exception, "test-only"):
            subject.bind_verified_panel(evidence)

    def test_registered_manifest_can_issue_exact_private_binding(self) -> None:
        evidence = _evidence(_panel())
        with mock.patch.object(
            subject,
            "REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256",
            evidence.composite_verifier_manifest_sha256,
        ):
            binding = subject.bind_verified_panel(evidence)
        binding.require_verified()
        self.assertEqual(binding.panel.canonical_sha256, _panel().canonical_sha256)

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

    def test_verifier_row_count_must_match_panel(self) -> None:
        with self.assertRaisesRegex(Exception, "ordered context count"):
            replace(_evidence(_panel()), observed_row_count=5)


class DiagnosticShapeTest(unittest.TestCase):
    def test_passing_fixture_is_complete_and_canonical(self) -> None:
        diagnostics = _diagnostics(_panel())
        self.assertEqual(len(diagnostics.observed_mode_q_bins), 72)
        self.assertIn((0, 5), diagnostics.observed_mode_q_bins)
        self.assertIn((11, 5), diagnostics.observed_mode_q_bins)
        self.assertEqual(len(diagnostics.canonical_sha256), 64)
        self.assertIsInstance(hash(diagnostics), int)

    def test_checkpoint_set_is_exact(self) -> None:
        with self.assertRaisesRegex(Exception, "0/100/250/500"):
            replace(
                _diagnostics(_panel()),
                evaluated_checkpoint_updates=(0, 100, 500),
            )

    def test_duplicate_mode_q_identity_is_refused(self) -> None:
        diagnostics = _diagnostics(_panel())
        with self.assertRaisesRegex(Exception, "duplicate"):
            replace(
                diagnostics,
                observed_mode_q_bins=(
                    diagnostics.observed_mode_q_bins[0],
                    diagnostics.observed_mode_q_bins[0],
                ),
            )

    def test_mode_counts_must_cover_panel(self) -> None:
        with self.assertRaisesRegex(Exception, "cover the panel"):
            replace(
                _diagnostics(_panel()),
                deterministic_mode_counts=(0,) * 12,
            )

    def test_all_sensitivity_features_are_required_exactly_once(self) -> None:
        diagnostics = _diagnostics(_panel())
        duplicate = diagnostics.sensitivity_diagnostics[:-1] + (
            diagnostics.sensitivity_diagnostics[0],
        )
        with self.assertRaisesRegex(Exception, "every registered feature"):
            replace(diagnostics, sensitivity_diagnostics=duplicate)

    def test_sensitivity_requires_estimate_inside_interval(self) -> None:
        with self.assertRaisesRegex(Exception, "inside"):
            _sensitivity(
                "scene_si", effect=0.5, lower=0.1, upper=0.3
            )


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

    def test_all_pass_test_fixture_still_cannot_authorize(self) -> None:
        assessment = self._assessment()
        self.assertTrue(assessment.all_gates_pass)
        self.assertFalse(assessment.continuation_authorized)
        self.assertIs(
            assessment.evidence_class,
            subject.EvidenceClass.TEST_ONLY_SYNTHETIC,
        )
        with self.assertRaises(subject.ContinuationRefused):
            assessment.require_continuation_authorized()

    def test_missing_highest_q_bin_fails_exact_coverage(self) -> None:
        observed = tuple(
            item for item in self.base.observed_mode_q_bins if item != (11, 5)
        )
        failed = self._failed(replace(self.base, observed_mode_q_bins=observed))
        self.assertIn("warmup_mode_q_coverage", failed)

    def test_success_and_failure_both_required(self) -> None:
        self.assertIn(
            "feedback_outcome_coverage",
            self._failed(replace(self.base, registered_failure_count=0)),
        )
        self.assertIn(
            "feedback_outcome_coverage",
            self._failed(replace(self.base, registered_success_count=0)),
        )

    def test_nonfinite_flag_fails(self) -> None:
        self.assertIn(
            "finite_numerics",
            self._failed(replace(self.base, all_numerics_finite=False)),
        )

    def test_rank_requires_strict_direction_not_numeric_threshold(self) -> None:
        failed = self._failed(
            replace(self.base, critic_rank_correlation_update500=0.1)
        )
        self.assertIn("critic_rank_direction", failed)
        tiny = replace(self.base, critic_rank_correlation_update500=0.1000001)
        self.assertNotIn("critic_rank_direction", self._failed(tiny))

    def test_action_regret_requires_strict_decrease(self) -> None:
        failed = self._failed(replace(self.base, action_regret_update500=0.3))
        self.assertIn("action_regret_direction", failed)

    def test_widest_support_mode_cannot_pin_every_context(self) -> None:
        counts = [0] * 12
        counts[11] = self.base.panel_context_count
        failed = self._failed(
            replace(self.base, deterministic_mode_counts=tuple(counts))
        )
        self.assertIn("no_widest_support_mode_pinning", failed)

    def test_continuous_q_regret_requires_strict_decrease(self) -> None:
        failed = self._failed(
            replace(self.base, continuous_q_regret_update500=0.2)
        )
        self.assertIn("continuous_q_regret_direction", failed)

    def test_sensitivity_needs_control_and_confidence_away_from_zero(self) -> None:
        original = self.base.sensitivity_diagnostics[0]
        inconclusive = replace(
            original,
            effect_estimate=0.01,
            confidence_interval_lower=-0.1,
            confidence_interval_upper=0.1,
        )
        diagnostics = replace(
            self.base,
            sensitivity_diagnostics=(inconclusive,)
            + self.base.sensitivity_diagnostics[1:],
        )
        self.assertIn("controlled_state_sensitivity", self._failed(diagnostics))
        uncontrolled = replace(original, only_registered_feature_changed=False)
        diagnostics = replace(
            self.base,
            sensitivity_diagnostics=(uncontrolled,)
            + self.base.sensitivity_diagnostics[1:],
        )
        self.assertIn("controlled_state_sensitivity", self._failed(diagnostics))

    def test_checkpoint_resume_must_be_bit_identical(self) -> None:
        failed = self._failed(
            replace(self.base, checkpoint_resume_bit_identical=False)
        )
        self.assertIn("checkpoint_resume_bit_identity", failed)

    def test_runtime_is_checked_against_prelaunch_panel_budget(self) -> None:
        failed = self._failed(replace(self.base, runtime_seconds=100.0001))
        self.assertIn("runtime_within_predeclared_bound", failed)
        self.assertNotIn(
            "runtime_within_predeclared_bound",
            self._failed(replace(self.base, runtime_seconds=100.0)),
        )

    def test_wrong_panel_seed_or_preregistration_is_refused_not_scored(self) -> None:
        with self.assertRaisesRegex(Exception, "another diagnostic panel"):
            self._assessment(replace(self.base, panel_sha256="4" * 64))
        with self.assertRaisesRegex(Exception, "seed 17"):
            self._assessment(replace(self.base, seed=29))
        with self.assertRaisesRegex(Exception, "another preregistration"):
            self._assessment(
                replace(self.base, preregistration_sha256="5" * 64)
            )

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
        assessment.require_continuation_authorized()

    def test_failed_verified_path_never_authorizes(self) -> None:
        evidence = _evidence(self.panel)
        with mock.patch.object(
            subject,
            "REGISTERED_COMPOSITE_VERIFIER_MANIFEST_SHA256",
            evidence.composite_verifier_manifest_sha256,
        ):
            binding = subject.bind_verified_panel(evidence)
        failed = replace(self.base, action_regret_update500=0.3)
        assessment = subject.assess_verified_smoke(binding, failed)
        self.assertFalse(assessment.continuation_authorized)
        with self.assertRaises(subject.ContinuationRefused):
            assessment.require_continuation_authorized()


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


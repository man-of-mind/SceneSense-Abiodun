"""Focused CPU-only tests for the Run-3 train-only preflight."""

from __future__ import annotations

import hashlib
import ast
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
from unittest import mock

import numpy as np

from . import run3_train_only_preflight as preflight_module
from .empirical_contextual_fit_partition import TRAIN_SPLIT
from .empirical_contextual_run3_reward import (
    RUN3_KERNEL_SPEC_SHA256,
    RUN3_REWARD_SPEC_SHA256,
)
from .modeled_smoke_support import MODELED_SMOKE_MODE_Q_E4_BOUNDS
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .run3_train_only_preflight import (
    EXPECTED_ACTIONS_PER_SCENE,
    EXPECTED_CONTEXT_PROFILE_ACTION_EVALUATIONS,
    EXPLICIT_Q_E4,
    FUTURE_DECISION_KEY_CONTRACT,
    QUALITY_FIELDS,
    RAW_QUALITY_FIELDS,
    SUMMARY_FIELDS,
    TAIL_SENSITIVITY_SPEC_SHA256,
    ActionVectorBatchV1,
    Run3PreflightError,
    Run3TrainOnlyPreflightV1,
    TrainContextV1,
    _conditional_timely_success_reward,
    _integrate_latency_proxy_vectors,
    render_run3_train_only_preflight,
)


class _SyntheticTrainSource:
    split = TRAIN_SPLIT
    reward_spec_sha256 = RUN3_REWARD_SPEC_SHA256
    kernel_spec_sha256 = RUN3_KERNEL_SPEC_SHA256
    source_bindings = {"synthetic_test_seam_sha256": "1" * 64}

    def __init__(
        self,
        *,
        count: int = 1,
        corrupt_expected: bool = False,
        zero_timely_profile: str | None = None,
        isolated_timely_profile: str | None = None,
    ) -> None:
        self._contexts = tuple(
            TrainContextV1(f"train-{index}", 1.0 + index / 10.0)
            for index in range(count)
        )
        self.corrupt_expected = corrupt_expected
        self.zero_timely_profile = zero_timely_profile
        self.isolated_timely_profile = isolated_timely_profile
        self.closed = False

    @property
    def contexts(self) -> Tuple[TrainContextV1, ...]:
        return self._contexts

    def close(self) -> None:
        self.closed = True

    def action_vectors(
        self, sample_id: str, network_profile: str, mode_id: int
    ) -> ActionVectorBatchV1:
        if sample_id not in {item.sample_id for item in self._contexts}:
            raise Run3PreflightError("synthetic foreign sample")
        profile_index = (
            "FAVORABLE_STABLE",
            "MID_VARIABLE",
            "ADVERSE_STABLE",
            "FADE_RECOVERY",
        ).index(network_profile)
        lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
        q = np.arange(lower, upper + 1, dtype=np.int64)
        qn = q.astype(np.float64) / 9800.0
        q_loc = np.clip(0.84 - 0.10 * qn - 0.001 * mode_id, 0.0, 1.0)
        q_seg = np.clip(0.90 - 0.48 * qn - 0.002 * mode_id, 0.0, 1.0)
        q_perc = q_loc * (0.7 + 0.3 * q_seg)
        payload = 410_000.0 * (1.0 - qn) + 12_000.0 + 100.0 * mode_id
        datagrams = np.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
        p_reassembly = np.clip(
            0.995 - 0.055 * profile_index - payload / 8.0e6, 0.05, 1.0
        )
        p_admission = np.clip(0.985 - 0.025 * profile_index + 0.0 * qn, 0.05, 1.0)
        p50 = 120.0 + 60.0 * (1.0 - qn) + 8.0 * profile_index
        p95 = p50 + 24.0
        p99 = p95 + 18.0
        if network_profile == self.zero_timely_profile:
            p50 = np.full(q.shape, 250.0, dtype=np.float64)
            p95 = np.full(q.shape, 275.0, dtype=np.float64)
            p99 = np.full(q.shape, 300.0, dtype=np.float64)
        if network_profile == self.isolated_timely_profile:
            p50 = np.full(q.shape, 250.0, dtype=np.float64)
            p95 = np.full(q.shape, 275.0, dtype=np.float64)
            p99 = np.full(q.shape, 300.0, dtype=np.float64)
            isolated = len(q) // 2
            p50[isolated] = 190.0
            p95[isolated] = 210.0
            p99[isolated] = 220.0
        conditional, timeout = _integrate_latency_proxy_vectors(
            q_perc, p50, p95, p99
        )
        p_admitted = p_reassembly * p_admission
        p_reassembly_failure = 1.0 - p_reassembly
        p_admission_failure = p_reassembly * (1.0 - p_admission)
        fail = p_reassembly_failure + p_admission_failure
        expected = -fail + p_admitted * conditional
        if self.corrupt_expected:
            expected = expected.copy()
            expected[0] += 0.1
        metrics: Dict[str, np.ndarray] = {
            "q_loc": q_loc,
            "q_seg": q_seg,
            "q_perc": q_perc,
            "vehicle_recall": np.clip(0.9 - 0.12 * qn, 0.0, 1.0),
            "person_recall": np.clip(0.82 - 0.08 * qn, 0.0, 1.0),
            "vehicle_xy_error_m": 0.5 + 0.2 * qn,
            "person_xy_error_m": 0.6 + 0.3 * qn,
            "seg_vehicle_iou": np.clip(0.92 - 0.3 * qn, 0.0, 1.0),
            "seg_person_iou": np.clip(0.65 - 0.35 * qn, 0.0, 1.0),
            "payload_bytes": payload,
            "datagram_count": datagrams.astype(np.float64),
            "p_complete_reassembly_given_sent": p_reassembly,
            "p_edge_admission_given_reassembled": p_admission,
            "p_reassembly_failure": p_reassembly_failure,
            "p_admission_failure": p_admission_failure,
            "p_admitted": p_admitted,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "p_timeout_given_admitted": timeout,
            "p_success_within_deadline_given_admitted": 1.0 - timeout,
            "p_service_timeout": p_admitted * timeout,
            "p_timely_feedback": p_admitted * (1.0 - timeout),
            "p_total_failure": fail + p_admitted * timeout,
            "conditional_admitted_expected_reward": conditional,
            "conditional_timely_success_reward": (
                _conditional_timely_success_reward(conditional, timeout)[0]
            ),
            "conditional_timely_success_reward_valid_fraction": (
                _conditional_timely_success_reward(conditional, timeout)[1]
            ),
            "expected_run3_return": expected,
        }
        for name in RAW_QUALITY_FIELDS:
            metrics[f"{name}_valid_fraction"] = np.ones(q.shape, dtype=np.float64)
        for label, multiplier in (("tail_stress_1p25", 1.25), ("tail_stress_1p50", 1.5)):
            stress_reward, stress_timeout = _integrate_latency_proxy_vectors(
                q_perc,
                p50,
                p95,
                p99,
                top_endpoint_multiplier=multiplier,
            )
            metrics[f"{label}_p_timeout_given_admitted"] = stress_timeout
            metrics[f"{label}_p_success_within_deadline_given_admitted"] = (
                1.0 - stress_timeout
            )
            metrics[f"{label}_p_service_timeout"] = p_admitted * stress_timeout
            metrics[f"{label}_p_timely_feedback"] = p_admitted * metrics[
                f"{label}_p_success_within_deadline_given_admitted"
            ]
            metrics[f"{label}_p_total_failure"] = (
                fail + p_admitted * stress_timeout
            )
            metrics[f"{label}_conditional_admitted_expected_reward"] = (
                stress_reward
            )
            (
                metrics[f"{label}_conditional_timely_success_reward"],
                metrics[
                    f"{label}_conditional_timely_success_reward_valid_fraction"
                ],
            ) = _conditional_timely_success_reward(stress_reward, stress_timeout)
            metrics[f"{label}_expected_run3_return"] = (
                -fail + p_admitted * stress_reward
            )
        batch = ActionVectorBatchV1(
            sample_id=sample_id,
            network_profile=network_profile,
            mode_id=mode_id,
            q_e4=q,
            metrics=metrics,
        )
        batch.revalidate()
        return batch


class TrainBoundaryTests(unittest.TestCase):
    def test_context_refuses_any_non_train_split(self) -> None:
        with self.assertRaisesRegex(Run3PreflightError, "only split='train'"):
            TrainContextV1("held-1", 1.0, split="held_scene")

    def test_engine_refuses_non_train_source(self) -> None:
        source = _SyntheticTrainSource()
        source.split = "fit_validation"
        with self.assertRaisesRegex(Run3PreflightError, "train-only"):
            Run3TrainOnlyPreflightV1(source, expected_scene_count=1)

    def test_engine_refuses_reward_and_kernel_binding_drift(self) -> None:
        reward = _SyntheticTrainSource()
        reward.reward_spec_sha256 = "0" * 64
        with self.assertRaisesRegex(Run3PreflightError, "reward binding"):
            Run3TrainOnlyPreflightV1(reward, expected_scene_count=1)
        kernel = _SyntheticTrainSource()
        kernel.kernel_spec_sha256 = "0" * 64
        with self.assertRaisesRegex(Run3PreflightError, "kernel binding"):
            Run3TrainOnlyPreflightV1(kernel, expected_scene_count=1)

    def test_scene_count_and_duplicate_identity_fail_closed(self) -> None:
        source = _SyntheticTrainSource(count=2)
        with self.assertRaisesRegex(Run3PreflightError, "scene count"):
            Run3TrainOnlyPreflightV1(source, expected_scene_count=1)
        source._contexts = (source._contexts[0], source._contexts[0])
        with self.assertRaisesRegex(Run3PreflightError, "duplicate"):
            Run3TrainOnlyPreflightV1(source, expected_scene_count=2)


class VectorContractTests(unittest.TestCase):
    def test_batch_has_exact_support_and_kernel_probabilities(self) -> None:
        source = _SyntheticTrainSource()
        batch = source.action_vectors("train-0", "ADVERSE_STABLE", 11)
        self.assertEqual(batch.q_e4[0], 0)
        self.assertEqual(batch.q_e4[-1], 9791)
        self.assertEqual(set(batch.metrics), set(SUMMARY_FIELDS))
        p_reassembly = batch.metrics["p_complete_reassembly_given_sent"]
        p_admission = batch.metrics["p_edge_admission_given_reassembled"]
        branch_sum = (
            (1.0 - p_reassembly)
            + p_reassembly * (1.0 - p_admission)
            + p_reassembly * p_admission
        )
        np.testing.assert_allclose(branch_sum, 1.0, rtol=0.0, atol=2e-15)

    def test_terminal_masses_and_success_reward_reconcile(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "ADVERSE_STABLE", 11
        )
        metrics = batch.metrics
        np.testing.assert_array_equal(
            metrics["p_service_timeout"],
            metrics["p_admitted"] * metrics["p_timeout_given_admitted"],
        )
        np.testing.assert_allclose(
            metrics["p_total_failure"] + metrics["p_timely_feedback"],
            1.0,
            rtol=0.0,
            atol=2e-15,
        )
        valid = metrics[
            "conditional_timely_success_reward_valid_fraction"
        ].astype(bool)
        reconstructed_conditional = (
            metrics["p_success_within_deadline_given_admitted"][valid]
            * metrics["conditional_timely_success_reward"][valid]
            - metrics["p_timeout_given_admitted"][valid]
        )
        np.testing.assert_allclose(
            reconstructed_conditional,
            metrics["conditional_admitted_expected_reward"][valid],
            rtol=0.0,
            atol=2e-15,
        )
        reconstructed_total = (
            -metrics["p_total_failure"][valid]
            + metrics["p_timely_feedback"][valid]
            * metrics["conditional_timely_success_reward"][valid]
        )
        np.testing.assert_allclose(
            reconstructed_total,
            metrics["expected_run3_return"][valid],
            rtol=0.0,
            atol=2e-15,
        )

    def test_zero_timely_success_reward_is_honestly_undefined(self) -> None:
        batch = _SyntheticTrainSource(
            zero_timely_profile="ADVERSE_STABLE"
        ).action_vectors("train-0", "ADVERSE_STABLE", 11)
        self.assertTrue(
            np.all(
                batch.metrics[
                    "conditional_timely_success_reward_valid_fraction"
                ]
                == 0.0
            )
        )
        self.assertTrue(
            np.all(np.isnan(batch.metrics["conditional_timely_success_reward"]))
        )
        np.testing.assert_array_equal(
            batch.metrics["conditional_admitted_expected_reward"],
            -batch.metrics["p_timeout_given_admitted"],
        )

    def test_terminal_metric_corruption_is_refused(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "FAVORABLE_STABLE", 11
        )
        for field in (
            "p_service_timeout",
            "p_total_failure",
            "conditional_timely_success_reward",
            "tail_stress_1p25_p_service_timeout",
            "tail_stress_1p50_p_total_failure",
        ):
            with self.subTest(field=field):
                metrics = {
                    key: values.copy() for key, values in batch.metrics.items()
                }
                metrics[field][0] += 0.01
                with self.assertRaisesRegex(
                    Run3PreflightError, "reconciliation|probability masses"
                ):
                    ActionVectorBatchV1(
                        batch.sample_id,
                        batch.network_profile,
                        batch.mode_id,
                        batch.q_e4,
                        metrics,
                    ).revalidate()

    def test_missing_raw_class_component_is_not_zero_imputed(self) -> None:
        source = _SyntheticTrainSource()
        batch = source.action_vectors("train-0", "FAVORABLE_STABLE", 11)
        metrics = dict(batch.metrics)
        values = metrics["person_xy_error_m"].copy()
        valid = metrics["person_xy_error_m_valid_fraction"].copy()
        values[7] = math.nan
        valid[7] = 0.0
        metrics["person_xy_error_m"] = values
        metrics["person_xy_error_m_valid_fraction"] = valid
        ActionVectorBatchV1(
            batch.sample_id, batch.network_profile, batch.mode_id, batch.q_e4, metrics
        ).revalidate()
        self.assertTrue(math.isnan(values[7]))
        self.assertEqual(valid[7], 0.0)

    def test_probability_and_datagram_corruption_are_refused(self) -> None:
        source = _SyntheticTrainSource()
        batch = source.action_vectors("train-0", "FAVORABLE_STABLE", 11)
        metrics = dict(batch.metrics)
        metrics["p_timely_feedback"] = np.ones_like(batch.q_e4, dtype=np.float64)
        with self.assertRaisesRegex(Run3PreflightError, "exact reconciliation"):
            ActionVectorBatchV1(
                batch.sample_id,
                batch.network_profile,
                batch.mode_id,
                batch.q_e4,
                metrics,
            ).revalidate()
        metrics = dict(batch.metrics)
        metrics["datagram_count"] = metrics["datagram_count"].copy()
        metrics["datagram_count"][0] += 1.0
        with self.assertRaisesRegex(Run3PreflightError, "datagram"):
            ActionVectorBatchV1(
                batch.sample_id,
                batch.network_profile,
                batch.mode_id,
                batch.q_e4,
                metrics,
            ).revalidate()

    def test_raw_validity_mask_must_exactly_match_finite_support(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "FAVORABLE_STABLE", 11
        )
        metrics = {name: values.copy() for name, values in batch.metrics.items()}
        metrics["person_xy_error_m"][3] = math.nan
        with self.assertRaisesRegex(Run3PreflightError, "validity mask"):
            ActionVectorBatchV1(
                batch.sample_id,
                batch.network_profile,
                batch.mode_id,
                batch.q_e4,
                metrics,
            ).revalidate()

    def test_raw_quality_physical_ranges_are_enforced(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "FAVORABLE_STABLE", 11
        )
        cases = (
            ("vehicle_recall", 1.01, r"escaped \[0,1\]"),
            ("seg_person_iou", -0.01, r"escaped \[0,1\]"),
            ("vehicle_xy_error_m", -0.01, "non-negative"),
        )
        for name, invalid, message in cases:
            with self.subTest(component=name):
                metrics = {
                    key: values.copy() for key, values in batch.metrics.items()
                }
                metrics[name][5] = invalid
                with self.assertRaisesRegex(Run3PreflightError, message):
                    ActionVectorBatchV1(
                        batch.sample_id,
                        batch.network_profile,
                        batch.mode_id,
                        batch.q_e4,
                        metrics,
                    ).revalidate()

    def test_latency_quantiles_must_be_nonnegative_and_ordered(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "FAVORABLE_STABLE", 11
        )
        cases = (
            ("latency_p50_ms", -0.01),
            ("latency_p95_ms", float(batch.metrics["latency_p50_ms"][4] - 0.01)),
            ("latency_p99_ms", float(batch.metrics["latency_p95_ms"][4] - 0.01)),
        )
        for name, invalid in cases:
            with self.subTest(quantile=name):
                metrics = {
                    key: values.copy() for key, values in batch.metrics.items()
                }
                metrics[name][4] = invalid
                with self.assertRaisesRegex(Run3PreflightError, "latency quantiles"):
                    ActionVectorBatchV1(
                        batch.sample_id,
                        batch.network_profile,
                        batch.mode_id,
                        batch.q_e4,
                        metrics,
                    ).revalidate()
    def test_tail_stress_is_monotone_and_analysis_only(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "ADVERSE_STABLE", 11
        )
        self.assertTrue(
            np.all(
                batch.metrics["tail_stress_1p50_expected_run3_return"]
                <= batch.metrics["tail_stress_1p25_expected_run3_return"] + 1e-15
            )
        )
        self.assertTrue(
            np.all(
                batch.metrics["tail_stress_1p25_expected_run3_return"]
                <= batch.metrics["expected_run3_return"] + 1e-15
            )
        )
        self.assertEqual(len(TAIL_SENSITIVITY_SPEC_SHA256), 64)


class EngineAndRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _SyntheticTrainSource()
        cls.result = Run3TrainOnlyPreflightV1(
            cls.source, expected_scene_count=1
        ).run()

    def test_compact_outputs_and_baselines_exist(self) -> None:
        summary = self.result.summary
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["scene_count"], 1)
        self.assertEqual(summary["materialized_context_action_rows"], 0)
        self.assertFalse(summary["run2_winner_or_checkpoint_reused"])
        self.assertFalse(summary["high_q_required_to_win"])
        self.assertTrue(summary["nonzero_timely_action_in_every_profile"])
        self.assertEqual(summary["reward_spec_sha256"], RUN3_REWARD_SPEC_SHA256)
        self.assertEqual(summary["kernel_spec_sha256"], RUN3_KERNEL_SPEC_SHA256)
        self.assertEqual(
            summary["future_training_decision_key_contract"],
            FUTURE_DECISION_KEY_CONTRACT,
        )
        self.assertEqual(
            [row["label"] for row in self.result.baseline_rows],
            [
                "TRAIN_SELECTED_FIXED_ACTION",
                "TRAIN_SELECTED_FIXED_MODE_CONTEXTUAL_BEST_Q",
                "TRAIN_CONTEXTUAL_MODE_AND_Q_ORACLE",
            ],
        )
        self.assertEqual(summary["mode_best_scalar_probe_count"], 4 * 12)
        self.assertEqual(summary["fixed_action_second_pass_scalar_probe_count"], 4)
        self.assertEqual(summary["fixed_action_second_pass_mode_vector_queries"], 4)
        self.assertLessEqual(summary["fixed_action_aggregate_max_abs_diff"], 1e-13)
        for label in ("tail_stress_1p25", "tail_stress_1p50"):
            self.assertEqual(summary["stress_mode_best_scalar_probe_count"][label], 48)
            self.assertEqual(
                summary["stress_contextual_winner_scalar_probe_count"][label], 4
            )
            self.assertEqual(
                summary["stress_fixed_action_scalar_probe_count"][label], 8
            )
            self.assertLessEqual(
                summary["stress_fixed_action_recheck_max_abs_diff"][label],
                1e-13,
            )
        for profile, region in summary["timely_region_gate"]["profile_maxima"].items():
            with self.subTest(profile=profile):
                self.assertGreaterEqual(region["count"], 2)
                self.assertGreaterEqual(region["q_e4_width"], 1)

    def test_quality_payload_kernel_and_latency_are_all_visible(self) -> None:
        row = self.result.baseline_rows[0]
        for name in SUMMARY_FIELDS:
            self.assertIn(name, row)
        self.assertIn("vehicle_recall_valid_fraction", row)
        self.assertIn("seg_person_iou_valid_fraction", row)

    def test_q_bins_and_explicit_q_points_are_visible(self) -> None:
        self.assertEqual(len(self.result.q_bin_rows), 3)
        self.assertEqual(
            {int(row["q_e4"]) for row in self.result.explicit_q_rows},
            set(EXPLICIT_Q_E4),
        )
        self.assertEqual(len(self.result.explicit_q_rows), 96)
        self.assertEqual(len(self.result.pareto_rows), 24)
        self.assertTrue(
            all(
                "pareto_qloc_vs_payload_within_mode" in row
                for row in self.result.pareto_rows
            )
        )
        profiles = {row["network_profile"] for row in self.result.contextual_winner_rows}
        self.assertEqual(
            profiles,
            {
                "FAVORABLE_STABLE",
                "MID_VARIABLE",
                "ADVERSE_STABLE",
                "FADE_RECOVERY",
            },
        )
        for profile in profiles:
            fraction = sum(
                row["weighted_fraction_within_profile"]
                for row in self.result.contextual_winner_rows
                if row["network_profile"] == profile
            )
            self.assertAlmostEqual(fraction, 1.0, places=15)
        exact_keys = {
            (row["network_profile"], row["mode_id"], row["q_e4"])
            for row in self.result.contextual_winner_rows
        }
        self.assertEqual(len(exact_keys), len(self.result.contextual_winner_rows))
        self.assertTrue(
            all(
                any(lower <= row["q_e4"] <= upper and name == row["q_bin"]
                    for name, lower, upper in preflight_module.Q_BINS)
                for row in self.result.contextual_winner_rows
            )
        )

    def test_tail_sensitivity_reports_ranking_robustness_without_gating(self) -> None:
        summary = self.result.summary
        self.assertIn(
            "DOES_NOT_AFFECT_GO_NO_GO",
            summary["tail_sensitivity_decision_role"],
        )
        for label, result in summary["tail_sensitivity_results"].items():
            with self.subTest(label=label):
                self.assertIn("COUNTERFACTUAL", result["analysis_role"])
                self.assertGreaterEqual(
                    result["contextual_base_winner_agreement_fraction"], 0.0
                )
                self.assertLessEqual(
                    result["contextual_base_winner_agreement_fraction"], 1.0
                )
                self.assertGreaterEqual(
                    result["contextual_stress_oracle_gain_over_base_winner"],
                    -1e-15,
                )
                self.assertGreaterEqual(
                    result["stress_optimal_fixed_action_gain_over_base_fixed"],
                    -1e-15,
                )
                self.assertEqual(
                    len(result["per_mode_contextual_best_q_under_stress"]), 12
                )
                self.assertGreaterEqual(
                    result["base_selected_mode_regret_under_stress"], -1e-15
                )
        self.assertTrue(
            any(
                not result["fixed_action_matches_stress_optimum"]
                for result in summary["tail_sensitivity_results"].values()
            )
        )

    def test_pareto_membership_depends_only_on_localization_and_payload(self) -> None:
        batch = _SyntheticTrainSource().action_vectors(
            "train-0", "FAVORABLE_STABLE", 11
        )
        original_metrics = dict(batch.metrics)
        original = Run3TrainOnlyPreflightV1._pareto_rows({11: original_metrics})

        changed_metrics = dict(original_metrics)
        changed_metrics["q_seg"] = original_metrics["q_seg"][::-1].copy()
        changed_metrics["q_perc"] = np.linspace(
            1.0, 0.0, len(batch.q_e4), dtype=np.float64
        )
        changed_metrics["latency_p50_ms"] = original_metrics[
            "latency_p50_ms"
        ][::-1].copy()
        changed = Run3TrainOnlyPreflightV1._pareto_rows({11: changed_metrics})

        original_flags = {
            int(row["q_e4"]): row["pareto_qloc_vs_payload_within_mode"]
            for row in original
        }
        changed_flags = {
            int(row["q_e4"]): row["pareto_qloc_vs_payload_within_mode"]
            for row in changed
        }
        self.assertEqual(original_flags, changed_flags)
        self.assertTrue(
            all(
                row["label"]
                == "HIGH_Q_LOCALIZATION_PAYLOAD_PARETO_VISIBILITY"
                for row in changed
            )
        )

    def test_scalar_vector_probe_is_decisive(self) -> None:
        authoritative_vector_integral = _integrate_latency_proxy_vectors

        def biased_vector_integral(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
            reward, timeout = authoritative_vector_integral(*args, **kwargs)
            return reward + 0.01, timeout

        with mock.patch.object(
            preflight_module,
            "_integrate_latency_proxy_vectors",
            side_effect=biased_vector_integral,
        ), mock.patch(
            __name__ + "._integrate_latency_proxy_vectors",
            side_effect=biased_vector_integral,
        ):
            source = _SyntheticTrainSource()
            with self.assertRaisesRegex(
                Run3PreflightError, "scalar/vector disagreement"
            ):
                Run3TrainOnlyPreflightV1(source, expected_scene_count=1).run()

    def test_independent_scalar_tail_probe_catches_tail_only_vector_bias(self) -> None:
        authoritative_vector_integral = _integrate_latency_proxy_vectors

        def tail_biased_integral(
            *args: Any, **kwargs: Any
        ) -> Tuple[np.ndarray, np.ndarray]:
            reward, timeout = authoritative_vector_integral(*args, **kwargs)
            if kwargs.get("top_endpoint_multiplier", 1.0) > 1.0:
                reward = reward + 0.01
            return reward, timeout

        with mock.patch.object(
            preflight_module,
            "_integrate_latency_proxy_vectors",
            side_effect=tail_biased_integral,
        ), mock.patch(
            __name__ + "._integrate_latency_proxy_vectors",
            side_effect=tail_biased_integral,
        ):
            with self.assertRaisesRegex(
                Run3PreflightError, "scalar/vector stress disagreement"
            ):
                Run3TrainOnlyPreflightV1(
                    _SyntheticTrainSource(), expected_scene_count=1
                ).run()

    def test_each_profile_requires_a_nonzero_timely_action(self) -> None:
        source = _SyntheticTrainSource(zero_timely_profile="ADVERSE_STABLE")
        with self.assertRaisesRegex(Run3PreflightError, "ADVERSE_STABLE"):
            Run3TrainOnlyPreflightV1(source, expected_scene_count=1).run()

    def test_isolated_single_timely_q_does_not_satisfy_region_gate(self) -> None:
        source = _SyntheticTrainSource(
            isolated_timely_profile="ADVERSE_STABLE"
        )
        with self.assertRaisesRegex(Run3PreflightError, "adjacent executable q pair"):
            Run3TrainOnlyPreflightV1(source, expected_scene_count=1).run()

    def test_rendering_is_byte_stable_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            hashes_a = render_run3_train_only_preflight(self.result, first)
            hashes_b = render_run3_train_only_preflight(self.result, second)
            self.assertEqual(hashes_a, hashes_b)
            names_a = sorted(path.name for path in first.iterdir())
            names_b = sorted(path.name for path in second.iterdir())
            self.assertEqual(names_a, names_b)
            for name in names_a:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertLess(
                sum(path.stat().st_size for path in first.iterdir()), 2_000_000
            )
            with self.assertRaises(FileExistsError):
                render_run3_train_only_preflight(self.result, first)

    def test_ties_are_deterministic(self) -> None:
        again = Run3TrainOnlyPreflightV1(
            _SyntheticTrainSource(), expected_scene_count=1
        ).run()
        self.assertEqual(self.result.canonical_sha256(), again.canonical_sha256())


class SourceAuditTests(unittest.TestCase):
    def test_module_has_no_training_cuda_or_random_sampling_import(self) -> None:
        import rl_agent.splitfusion_hybrid_sac_v1.run3_train_only_preflight as module

        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", imported)
        self.assertNotIn("Run3ExpectedOutcomeV1(", source)
        self.assertNotIn("sample_run3_simulator_outcome", source)

    def test_estimate_discloses_streaming_and_bounded_disk(self) -> None:
        estimate = Run3TrainOnlyPreflightV1.estimate_full_run()
        self.assertEqual(estimate["materialized_context_action_rows"], 0)
        self.assertEqual(EXPECTED_ACTIONS_PER_SCENE, 52_240)
        self.assertEqual(EXPECTED_CONTEXT_PROFILE_ACTION_EVALUATIONS, 81_703_360)
        self.assertEqual(estimate["action_evaluations"], 81_703_360)
        self.assertLessEqual(estimate["estimated_output_bytes_upper_bound"], 2_000_000)

    def test_documentation_distinguishes_legacy_held_and_fit_validation(self) -> None:
        source = inspect.getsource(preflight_module)
        self.assertIn("legacy ``held_scene``", source)
        self.assertIn("``fit_validation`` identities", source)
        self.assertIn("neither legacy ``held_scene`` nor", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

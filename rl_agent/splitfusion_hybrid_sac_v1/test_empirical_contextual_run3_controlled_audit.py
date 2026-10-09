from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

import numpy as np

from .empirical_contextual_fit_validation_evaluator import (
    FitValidationActorEvaluatorV1,
)
from .empirical_contextual_run3_controlled_audit import (
    AUDIT_CLASS,
    AUDIT_SCOPE_DISCLOSURE,
    AUDIT_UPDATES,
    FIXED_COMPARATOR,
    ActionAuditV1,
    ComponentAuditV1,
    ControlledAuditError,
    ControlledAuditRowV1,
    _assert_panel_scope,
    _candidate_key,
    _oracle_action,
    _validated_regret,
    _verify_fixed_comparator,
    aggregate_controlled_rows,
    aggregate_paired_progress,
    expected_run3_reward_vector,
)
from .empirical_contextual_run3_reward import (
    QuantileLatencyProxyV1,
    expected_run3_reward,
)
from .empirical_contextual_split_oracle import (
    EXPECTED_EXACT_ACTION_COUNT_PER_SCENE,
)


def _component(value: float | None, *, valid: bool = True) -> ComponentAuditV1:
    return ComponentAuditV1(
        value=value,
        valid=valid,
        status="VALID" if valid else "NO_QUALIFIED_GT",
    )


def _action(
    *,
    reward: float,
    q_perc: float = 0.6,
    latency: float = 180.0,
    person_valid: bool = True,
) -> ActionAuditV1:
    quality = {
        "seg_vehicle_iou": _component(0.7),
        "seg_person_iou": _component(0.5 if person_valid else None, valid=person_valid),
        "vehicle_recall": _component(0.8),
        "person_recall": _component(0.4 if person_valid else None, valid=person_valid),
        "vehicle_xy_error_m": _component(0.7),
        "person_xy_error_m": _component(1.2 if person_valid else None, valid=person_valid),
        "q_seg": _component(0.62),
        "q_loc": _component(0.58),
        "q_perc": _component(q_perc),
    }
    return ActionAuditV1(
        mode_id=11,
        q_e4=6095,
        requested_q=None,
        family="split_ae32",
        quantizer="UINT8",
        **quality,
        total_transmitted_bytes=2_800.0,
        datagram_count=1,
        p_complete_reassembly_given_sent=0.98,
        p_edge_admission_given_reassembled=0.97,
        p_edge_admission_given_sent=0.9506,
        conditional_feature_uplink_p50_ms=67.0,
        conditional_feature_uplink_p95_ms=87.0,
        conditional_feature_uplink_p99_ms=107.0,
        fixed_stage_latency_ms=113.0,
        total_latency_p50_ms=latency,
        total_latency_p95_ms=latency + 20.0,
        total_latency_p99_ms=latency + 40.0,
        budget_miss_p50=latency > 200.0,
        budget_miss_p95=latency + 20.0 > 200.0,
        budget_miss_p99=latency + 40.0 > 200.0,
        expected_reward=reward,
        p_reassembly_failure=0.02,
        p_admission_failure=0.0294,
        p_admitted=0.9506,
        p_timeout_given_admitted=0.1,
        p_timely_feedback=0.85554,
        conditional_admitted_expected_reward=0.3,
    )


def _row(
    *, seed: int, update: int, panel: int, reward: float,
    oracle_reward: float = 0.5, fixed_reward: float = 0.2,
    person_valid: bool = True,
) -> ControlledAuditRowV1:
    learned = _action(reward=reward, person_valid=person_valid)
    fixed = _action(reward=fixed_reward)
    oracle = _action(reward=oracle_reward)
    return ControlledAuditRowV1(
        seed=seed,
        update_index=update,
        actor_state_sha256="a" * 64,
        checkpoint_content_sha256="b" * 64,
        checkpoint_file_sha256="c" * 64,
        panel_index=panel,
        scene_rank=panel,
        sample_id=f"sample-{panel}",
        episode_id="episode",
        frame_id=900 + panel * 4,
        network_profile="FAVORABLE_STABLE",
        radio_csv_row_number=panel,
        radio_trace_step_index=panel,
        camera_si=0.4,
        radar_p40=0.3,
        learned=learned,
        fixed=fixed,
        oracle=oracle,
        learned_oracle_regret=oracle_reward - reward,
        fixed_oracle_regret=oracle_reward - fixed_reward,
        learned_minus_fixed_reward=reward - fixed_reward,
    )


class Run3ControlledAuditTests(unittest.TestCase):
    def test_scope_is_explicitly_not_temporal_or_live(self) -> None:
        self.assertEqual(
            AUDIT_CLASS, "INTERNAL_CONTROLLED_OFFLINE_AUDIT_NOT_TEMPORAL_OR_LIVE"
        )
        self.assertIn("not consecutive route frames", AUDIT_SCOPE_DISCLOSURE)
        self.assertIn("not live deployment", AUDIT_SCOPE_DISCLOSURE)
        self.assertEqual(AUDIT_UPDATES, (0, 5_000, 10_000))
        self.assertEqual(FIXED_COMPARATOR, (11, 6095))

    def test_vector_expectation_matches_authoritative_scalar(self) -> None:
        quality = np.asarray([0.35, 0.55, 0.75, 0.9], dtype=np.float64)
        p_reassembly = np.asarray([0.6, 0.8, 0.95, 1.0], dtype=np.float64)
        p_admission = np.asarray([0.5, 0.9, 0.7, 1.0], dtype=np.float64)
        p50 = np.asarray([150.0, 190.0, 205.0, 100.0], dtype=np.float64)
        p95 = np.asarray([180.0, 230.0, 230.0, 210.0], dtype=np.float64)
        p99 = np.asarray([220.0, 280.0, 300.0, 260.0], dtype=np.float64)
        vector = expected_run3_reward_vector(
            q_perc=quality,
            p_reassembly=p_reassembly,
            p_admission_given_reassembled=p_admission,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
        )
        scalar = np.asarray([
            expected_run3_reward(
                q_perc=float(quality[index]),
                p_complete_reassembly_given_sent=float(p_reassembly[index]),
                p_edge_admission_given_reassembled=float(p_admission[index]),
                latency_proxy=QuantileLatencyProxyV1(
                    float(p50[index]), float(p95[index]), float(p99[index])
                ),
            ).expected_reward
            for index in range(len(quality))
        ])
        np.testing.assert_allclose(vector, scalar, rtol=0.0, atol=2e-15)

    def test_vector_refuses_shape_and_latency_drift(self) -> None:
        ones = np.ones(2)
        with self.assertRaises(ControlledAuditError):
            expected_run3_reward_vector(
                q_perc=ones,
                p_reassembly=ones[:1],
                p_admission_given_reassembled=ones,
                latency_p50_ms=100.0 * ones,
                latency_p95_ms=120.0 * ones,
                latency_p99_ms=140.0 * ones,
            )
        with self.assertRaises(ControlledAuditError):
            expected_run3_reward_vector(
                q_perc=ones,
                p_reassembly=ones,
                p_admission_given_reassembled=ones,
                latency_p50_ms=150.0 * ones,
                latency_p95_ms=120.0 * ones,
                latency_p99_ms=140.0 * ones,
            )

    def test_missing_quality_is_not_coerced_to_zero(self) -> None:
        component = _component(None, valid=False)
        self.assertIsNone(component.value)
        with self.assertRaises(ControlledAuditError):
            ComponentAuditV1(value=0.0, valid=False, status="MISSING")
        with self.assertRaises(ControlledAuditError):
            ComponentAuditV1(value=None, valid=True, status="VALID")

    def test_flat_row_carries_values_validity_and_status(self) -> None:
        flattened = _row(
            seed=17, update=0, panel=0, reward=0.1, person_valid=False
        ).flat()
        self.assertIsNone(flattened["learned_seg_person_iou"])
        self.assertFalse(flattened["learned_seg_person_iou_valid"])
        self.assertEqual(
            flattened["learned_seg_person_iou_status"], "NO_QUALIFIED_GT"
        )
        self.assertIn("learned_person_xy_error_m_valid", flattened)
        self.assertIn("learned_q_perc", flattened)

    def test_aggregates_report_component_denominators(self) -> None:
        rows = (
            _row(seed=17, update=0, panel=0, reward=0.1, person_valid=False),
            _row(seed=17, update=0, panel=1, reward=0.3, person_valid=True),
        )
        summaries = aggregate_controlled_rows(
            rows, require_registered_coverage=False
        )
        summary = next(
            row for row in summaries
            if row["network_profile"] == "ALL_PROFILES"
        )
        self.assertAlmostEqual(summary["expected_reward_mean"], 0.2)
        self.assertEqual(summary["seg_vehicle_iou_valid_count"], 2)
        self.assertEqual(summary["seg_person_iou_valid_count"], 1)
        self.assertEqual(summary["person_xy_error_m_valid_count"], 1)

    def test_paired_progress_is_matched_by_panel(self) -> None:
        rows = (
            _row(seed=17, update=0, panel=0, reward=0.1),
            _row(seed=17, update=0, panel=1, reward=0.2),
            _row(seed=17, update=5_000, panel=0, reward=0.2),
            _row(seed=17, update=5_000, panel=1, reward=0.2),
            _row(seed=17, update=10_000, panel=0, reward=0.3),
            _row(seed=17, update=10_000, panel=1, reward=0.1),
        )
        summaries = aggregate_paired_progress(
            rows, require_registered_coverage=False
        )
        five_to_ten = next(
            row for row in summaries
            if row["start_update"] == 5_000 and row["end_update"] == 10_000
        )
        self.assertEqual(five_to_ten["matched_context_count"], 2)
        self.assertEqual(five_to_ten["improved_context_count"], 1)
        self.assertEqual(five_to_ten["worsened_context_count"], 1)
        self.assertAlmostEqual(five_to_ten["expected_reward_delta_mean"], 0.0)

    def test_action_deadline_flags_are_exactly_200_ms(self) -> None:
        action = _action(reward=0.1, latency=180.0)
        self.assertFalse(action.budget_miss_p50)
        self.assertFalse(action.budget_miss_p95)  # equality at 200 ms is timely
        self.assertTrue(action.budget_miss_p99)

    def test_audit_source_does_not_use_historical_d1_reward(self) -> None:
        source = inspect.getsource(
            __import__(
                "rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_run3_controlled_audit",
                fromlist=["*"],
            )
        )
        self.assertNotIn("PILOT_UTILITY_SPEC", source)
        self.assertIn("expected_run3_reward", source)


    def test_registered_oracle_tie_key_is_lexicographic(self) -> None:
        baseline = _candidate_key(0.4, 0.8, 180.0, 5, 5000)
        cases = (
            _candidate_key(0.5, 0.0, 999.0, 11, 9800),
            _candidate_key(0.4, 0.9, 999.0, 11, 9800),
            _candidate_key(0.4, 0.8, 170.0, 11, 9800),
            _candidate_key(0.4, 0.8, 180.0, 4, 9800),
            _candidate_key(0.4, 0.8, 180.0, 5, 4999),
        )
        for preferred in cases:
            with self.subTest(preferred=preferred):
                self.assertGreater(preferred, baseline)

    def test_regret_check_is_not_vacuous(self) -> None:
        self.assertAlmostEqual(_validated_regret(0.5, 0.4), 0.1)
        self.assertEqual(
            _validated_regret(0.5, 0.5 + 0.5 * 3e-9),
            0.0,
        )
        with self.assertRaises(ControlledAuditError):
            _validated_regret(0.5, 0.500001)

    def test_real_registered_panel_scope_closes(self) -> None:
        with FitValidationActorEvaluatorV1() as evaluator:
            _assert_panel_scope(evaluator)
            self.assertEqual(len(evaluator.panel.entries), 340)
            self.assertEqual(
                len(
                    {
                        (
                            row.scene_sample_id,
                            row.scene_episode_id,
                            row.scene_frame_id,
                        )
                        for row in evaluator.panel.entries
                    }
                ),
                85,
            )

    def test_fixed_comparator_is_bound_to_train_only_preflight(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        binding = _verify_fixed_comparator(project_root)
        self.assertEqual(binding["label"], "TRAIN_SELECTED_FIXED_ACTION")
        self.assertEqual((binding["mode_id"], binding["q_e4"]), (11, 6095))
        self.assertEqual(
            binding["preflight_result_file_sha256"],
            "52801ce029eaac4dfde4834a2d8031429663a3efb383deed7ec1ed06d50ebab7",
        )
    def test_one_real_panel_oracle_is_scalar_revalidated(self) -> None:
        # This is bounded to one context (52,240 vector evaluations), launches
        # no CARLA/OAI/CUDA service, and exercises the real registered surfaces.
        with FitValidationActorEvaluatorV1() as evaluator:
            oracle = _oracle_action(evaluator, evaluator.panel.entries[0])
        self.assertTrue(math.isfinite(oracle.expected_reward))
        self.assertGreaterEqual(oracle.q_e4, 0)
        self.assertLessEqual(oracle.q_e4, 9800)
        self.assertEqual(
            EXPECTED_EXACT_ACTION_COUNT_PER_SCENE,
            sum(upper - lower + 1 for lower, upper in __import__(
                "rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support",
                fromlist=["MODELED_SMOKE_MODE_Q_E4_BOUNDS"],
            ).MODELED_SMOKE_MODE_Q_E4_BOUNDS),
        )


if __name__ == "__main__":
    unittest.main()

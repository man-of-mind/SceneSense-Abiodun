from __future__ import annotations

import inspect
import math
import random
import unittest
from dataclasses import fields, replace

from rl_agent.splitfusion_hybrid_sac_run4_v1.exploration import (
    ActionTraceSample,
    ActorPathError,
    CoverageGateConfig,
    CoverageObservation,
    CoverageRecordError,
    DecisionPhase,
    ExplorationCoverageLedger,
    ExplorationError,
    GradientStartRefused,
    OutcomeMetricThreshold,
    PreviousDecisionObservation,
    SelectionKind,
    StateFeatureThreshold,
    StratifiedWarmupSchedule,
    WarmupScheduleConfig,
    partition_quality_support,
    require_selection_path,
    summarize_action_trace,
)
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    round_half_up_q_e4,
)


def _bounds(lower: int = 0, upper: int = 99) -> tuple[tuple[int, int], ...]:
    return tuple((lower + mode, upper + mode) for mode in range(EXPECTED_MODE_COUNT))


def _schedule(
    *, bins: int = 4, samples: int = 2, seed: int = 17
) -> StratifiedWarmupSchedule:
    return StratifiedWarmupSchedule(
        WarmupScheduleConfig(
            mode_q_e4_bounds=_bounds(),
            q_bin_count=bins,
            samples_per_q_bin=samples,
            master_seed=seed,
            support_contract_id="unit-test-support-v1",
        )
    )


def _state_threshold(
    name: str,
    *,
    lower: float,
    upper: float,
    maximum_saturation: float = 0.0,
) -> StateFeatureThreshold:
    return StateFeatureThreshold(
        name=name,
        min_finite_count=2,
        min_unique_values=2,
        min_span=0.01,
        min_per_mode_finite_count=2,
        min_per_mode_unique_values=2,
        min_per_mode_span=0.01,
        saturation_lower=lower,
        saturation_upper=upper,
        saturation_tolerance=0.0,
        max_boundary_saturation_fraction=maximum_saturation,
    )


def _gate(*, bins: int = 4, samples: int = 2) -> CoverageGateConfig:
    return CoverageGateConfig(
        preregistration_id="provisional-test-gate-v1",
        min_current_actions_per_mode=bins * samples,
        min_current_actions_per_q_bin=samples,
        min_previous_actions_per_mode=bins * samples,
        min_previous_actions_per_q_bin=samples,
        min_previous_present=EXPECTED_MODE_COUNT * bins * samples,
        min_previous_success=1,
        min_previous_failure=1,
        state_thresholds=(
            _state_threshold("scene_si", lower=-1000.0, upper=10000.0),
            _state_threshold("scene_p40", lower=-1.0, upper=2.0),
            _state_threshold("prior_ul_mcs_index", lower=-1.0, upper=29.0),
            _state_threshold("rlc_backlog_bytes", lower=-1.0, upper=1e9),
        ),
        outcome_thresholds=(
            OutcomeMetricThreshold(
                name="previous_quality",
                min_finite_count=2,
                min_unique_values=2,
                min_span=0.001,
            ),
            OutcomeMetricThreshold(
                name="previous_latency_ms",
                min_finite_count=2,
                min_unique_values=2,
                min_span=0.1,
            ),
        ),
    )


def _observation(
    schedule: StratifiedWarmupSchedule,
    state_ordinal: int,
    *,
    constant_feature: str | None = None,
    saturated_feature: str | None = None,
    all_success: bool = False,
    constant_quality: bool = False,
    nonfinite_feature: str | None = None,
) -> CoverageObservation:
    previous = None
    if state_ordinal:
        action = schedule.action_at(state_ordinal - 1)
        success = True if all_success else state_ordinal % 2 == 0
        previous = PreviousDecisionObservation(
            mode_id=action.mode_id,
            q_e4=action.q_e4,
            success=success,
            quality=(
                (0.75 if constant_quality else 0.5 + state_ordinal / 1000.0)
                if success
                else None
            ),
            latency_ms=(40.0 + state_ordinal if success else None),
        )
    values = {
        "scene_si": 10.0 + state_ordinal,
        "scene_p40": 0.2 + (state_ordinal % 11) / 20.0,
        "prior_ul_mcs_index": state_ordinal % 29,
        "rlc_backlog_bytes": 100.0 + 37.0 * state_ordinal,
    }
    if constant_feature is not None:
        values[constant_feature] = (
            10 if constant_feature == "prior_ul_mcs_index" else 0.25
        )
    if saturated_feature == "scene_p40":
        values[saturated_feature] = 0.0
    if nonfinite_feature is not None:
        values[nonfinite_feature] = math.nan
    return CoverageObservation(previous=previous, **values)


def _filled_ledger(
    *,
    constant_feature: str | None = None,
    saturated_feature: str | None = None,
    all_success: bool = False,
    constant_quality: bool = False,
    nonfinite_feature: str | None = None,
    gate: CoverageGateConfig | None = None,
) -> ExplorationCoverageLedger:
    schedule = _schedule()
    ledger = ExplorationCoverageLedger(schedule, gate or _gate())
    for ordinal, action in enumerate(schedule.actions):
        ledger.record_decision(
            decision_identity=f"decision-{ordinal}",
            action=action,
            observation=_observation(
                schedule,
                ordinal,
                constant_feature=constant_feature,
                saturated_feature=saturated_feature,
                all_success=all_success,
                constant_quality=constant_quality,
                nonfinite_feature=nonfinite_feature,
            ),
        )
    ledger.record_final_feedback_state(
        _observation(
            schedule,
            len(schedule),
            constant_feature=constant_feature,
            saturated_feature=saturated_feature,
            all_success=all_success,
            constant_quality=constant_quality,
            nonfinite_feature=nonfinite_feature,
        )
    )
    return ledger


class StratifiedScheduleTests(unittest.TestCase):
    def test_every_mode_and_q_bin_is_covered_and_interleaved(self) -> None:
        schedule = _schedule(bins=5, samples=3)
        self.assertEqual(len(schedule), 12 * 5 * 3)
        for block_start in range(0, len(schedule), EXPECTED_MODE_COUNT):
            block = schedule.actions[block_start : block_start + EXPECTED_MODE_COUNT]
            self.assertEqual({item.mode_id for item in block}, set(range(12)))

        for mode_id in range(12):
            rows = [item for item in schedule.actions if item.mode_id == mode_id]
            self.assertEqual(len(rows), 15)
            self.assertEqual(
                [sum(item.q_bin_index == bin_index for item in rows) for bin_index in range(5)],
                [3] * 5,
            )
            lower, upper = schedule.config.mode_q_e4_bounds[mode_id]
            self.assertEqual(min(item.q_e4 for item in rows), lower)
            self.assertEqual(max(item.q_e4 for item in rows), upper)
            for item in rows:
                self.assertLessEqual(item.q_bin_lower_e4, item.q_e4)
                self.assertLessEqual(item.q_e4, item.q_bin_upper_e4)
                self.assertEqual(round_half_up_q_e4(item.requested_q), item.q_e4)

    def test_reproducibility_seed_difference_and_global_rng_isolation(self) -> None:
        random.seed(123456)
        before = random.getstate()
        first = _schedule(seed=91)
        after_first = random.getstate()
        second = _schedule(seed=91)
        after_second = random.getstate()
        different = _schedule(seed=92)
        self.assertEqual(before, after_first)
        self.assertEqual(before, after_second)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.config.schedule_id, second.config.schedule_id)
        self.assertNotEqual(first.actions, different.actions)
        self.assertNotEqual(first.config.schedule_id, different.config.schedule_id)
        self.assertEqual(
            len({item.counter_identity for item in first.actions}), len(first)
        )

    def test_partition_bounds_balance_and_half_up_tie(self) -> None:
        bins = partition_quality_support(0, 9, 3)
        self.assertEqual(
            [(item.lower_e4, item.upper_e4) for item in bins],
            [(0, 2), (3, 5), (6, 9)],
        )
        self.assertLessEqual(
            max(item.point_count for item in bins)
            - min(item.point_count for item in bins),
            1,
        )
        tie_schedule = StratifiedWarmupSchedule(
            WarmupScheduleConfig(
                mode_q_e4_bounds=tuple((0, 1) for _ in range(12)),
                q_bin_count=1,
                samples_per_q_bin=3,
                master_seed=1,
                support_contract_id="two-point-support",
            )
        )
        self.assertEqual(
            sorted(item.q_e4 for item in tie_schedule.actions if item.mode_id == 0),
            [0, 1, 1],
        )
        self.assertEqual(tie_schedule.classify_q(0, 0), 0)
        self.assertEqual(tie_schedule.classify_q(0, 1), 0)
        with self.assertRaises(ExplorationError):
            tie_schedule.classify_q(0, 2)
        with self.assertRaises(ExplorationError):
            partition_quality_support(0, 1, 3)

    def test_bounds_are_constructor_supplied_not_catalog_anchor_constants(self) -> None:
        signature = inspect.signature(WarmupScheduleConfig)
        self.assertIs(signature.parameters["mode_q_e4_bounds"].default, inspect.Parameter.empty)
        custom = StratifiedWarmupSchedule(
            WarmupScheduleConfig(
                mode_q_e4_bounds=tuple((100 * mode, 100 * mode + 49) for mode in range(12)),
                q_bin_count=5,
                samples_per_q_bin=2,
                master_seed=4,
                support_contract_id="custom-support",
            )
        )
        for action in custom.actions:
            lower, upper = custom.config.mode_q_e4_bounds[action.mode_id]
            self.assertLessEqual(lower, action.q_e4)
            self.assertLessEqual(action.q_e4, upper)


class ObservationContractTests(unittest.TestCase):
    def test_changed_coverage_gate_has_v2_record_identity(self) -> None:
        self.assertEqual(
            _gate().to_canonical_dict()["record"],
            "run4_exploration_coverage_gate_v2",
        )

    def test_only_required_state_and_previous_outcome_are_accepted(self) -> None:
        raw = {
            "scene_si": 12.0,
            "scene_p40": 0.4,
            "prior_ul_mcs_index": 9,
            "rlc_backlog_bytes": 1200,
            "previous": {
                "mode_id": 2,
                "q_e4": 7000,
                "success": True,
                "quality": 0.8,
                "latency_ms": 95.0,
            },
        }
        observation = CoverageObservation.from_mapping(raw)
        self.assertTrue(observation.previous_present)
        self.assertEqual(observation.previous.mode_id, 2)  # type: ignore[union-attr]
        self.assertNotIn("reward", {item.name for item in fields(CoverageObservation)})
        self.assertNotIn("reward", {item.name for item in fields(PreviousDecisionObservation)})

        for forbidden_key in ("network_profile", "profile_id", "frame_id", "previous_reward"):
            poisoned = dict(raw)
            poisoned[forbidden_key] = "forbidden"
            with self.subTest(forbidden_key=forbidden_key), self.assertRaises(
                CoverageRecordError
            ):
                CoverageObservation.from_mapping(poisoned)

    def test_nonfinite_values_are_retained_for_fail_closed_gate(self) -> None:
        observation = CoverageObservation(
            scene_si=math.nan,
            scene_p40=0.2,
            prior_ul_mcs_index=5,
            rlc_backlog_bytes=0,
            previous=None,
        )
        self.assertTrue(math.isnan(observation.scene_si))

    def test_prior_ul_mcs_requires_an_exact_table_zero_index(self) -> None:
        for raw in (0, 28):
            with self.subTest(raw=raw):
                observation = CoverageObservation(
                    scene_si=1.0,
                    scene_p40=0.2,
                    prior_ul_mcs_index=raw,
                    rlc_backlog_bytes=0,
                    previous=None,
                )
                self.assertEqual(observation.prior_ul_mcs_index, raw)

        for raw in (-1, 29, 12.0, True):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                CoverageRecordError, "exact table-0 index"
            ):
                CoverageObservation(
                    scene_si=1.0,
                    scene_p40=0.2,
                    prior_ul_mcs_index=raw,  # type: ignore[arg-type]
                    rlc_backlog_bytes=0,
                    previous=None,
                )

    def test_malformed_previous_outcome_semantics_are_rejected(self) -> None:
        common = {"mode_id": 0, "q_e4": 100}
        invalid = (
            {**common, "success": True, "quality": None, "latency_ms": 20.0},
            {**common, "success": True, "quality": 0.8, "latency_ms": None},
            {**common, "success": True, "quality": math.nan, "latency_ms": 20.0},
            {**common, "success": True, "quality": 1.01, "latency_ms": 20.0},
            {**common, "success": True, "quality": 0.8, "latency_ms": math.inf},
            {**common, "success": True, "quality": 0.8, "latency_ms": 170.001},
            {**common, "success": False, "quality": 0.8, "latency_ms": None},
            {**common, "success": False, "quality": None, "latency_ms": 170.0},
        )
        for fields_ in invalid:
            with self.subTest(fields=fields_), self.assertRaises(CoverageRecordError):
                PreviousDecisionObservation(**fields_)
        accepted = PreviousDecisionObservation(
            **common, success=True, quality=0.0, latency_ms=170.0
        )
        self.assertEqual(accepted.latency_ms, 170.0)
        failed = PreviousDecisionObservation(
            **common, success=False, quality=None, latency_ms=None
        )
        self.assertFalse(failed.success)


class FeedbackCoverageGateTests(unittest.TestCase):
    def test_complete_real_sequence_passes_all_current_and_previous_strata(self) -> None:
        ledger = _filled_ledger()
        report = ledger.require_gradient_start()
        self.assertTrue(report.gradient_start_allowed)
        self.assertTrue(report.final_feedback_state_recorded)
        self.assertEqual(report.decision_count, report.expected_decision_count)
        self.assertEqual(
            report.current_action_diagnostics.mode_counts,
            (8,) * EXPECTED_MODE_COUNT,
        )
        self.assertEqual(report.previous_mode_counts, (8,) * EXPECTED_MODE_COUNT)
        self.assertTrue(
            all(row == (2, 2, 2, 2) for row in report.previous_q_bin_counts)
        )
        self.assertEqual(report.previous_absent_count, 1)
        self.assertEqual(report.previous_present_count, len(ledger.schedule))
        self.assertGreater(report.previous_success_count, 0)
        self.assertGreater(report.previous_failure_count, 0)
        self.assertGreater(report.outcome_stat("previous_quality").span or 0.0, 0.0)
        self.assertGreater(report.outcome_stat("previous_latency_ms").span or 0.0, 0.0)

    def test_no_gradient_before_all_modes_bins_and_final_feedback(self) -> None:
        schedule = _schedule()
        ledger = ExplorationCoverageLedger(schedule, _gate())
        for ordinal, action in enumerate(schedule.actions[:-1]):
            ledger.record_decision(
                decision_identity=f"decision-{ordinal}",
                action=action,
                observation=_observation(schedule, ordinal),
            )
        report = ledger.report()
        self.assertFalse(report.gradient_start_allowed)
        self.assertTrue(any("warm-up incomplete" in item for item in report.failures))
        self.assertTrue(any("q bin" in item for item in report.failures))
        with self.assertRaises(GradientStartRefused):
            ledger.require_gradient_start()

        last_ordinal = len(schedule) - 1
        ledger.record_decision(
            decision_identity=f"decision-{last_ordinal}",
            action=schedule.action_at(last_ordinal),
            observation=_observation(schedule, last_ordinal),
        )
        report = ledger.report()
        self.assertFalse(report.gradient_start_allowed)
        self.assertIn(
            "final scheduled action has no successor feedback state", report.failures
        )

    def test_previous_action_link_and_counter_order_fail_closed(self) -> None:
        schedule = _schedule()
        ledger = ExplorationCoverageLedger(schedule, _gate())
        with self.assertRaises(CoverageRecordError):
            ledger.record_decision(
                decision_identity="out-of-order",
                action=schedule.action_at(1),
                observation=_observation(schedule, 0),
            )
        ledger.record_decision(
            decision_identity="decision-0",
            action=schedule.action_at(0),
            observation=_observation(schedule, 0),
        )
        with self.assertRaises(CoverageRecordError):
            ledger.record_decision(
                decision_identity="decision-1",
                action=schedule.action_at(1),
                observation=CoverageObservation(
                    scene_si=1,
                    scene_p40=0.2,
                    prior_ul_mcs_index=3,
                    rlc_backlog_bytes=4,
                    previous=None,
                ),
            )
        with self.assertRaises(CoverageRecordError):
            ledger.record_decision(
                decision_identity="decision-0",
                action=schedule.action_at(1),
                observation=_observation(schedule, 1),
            )

    def test_constant_saturated_and_nonfinite_state_evidence_refuses_gradient(self) -> None:
        constant = _filled_ledger(constant_feature="prior_ul_mcs_index").report()
        self.assertFalse(constant.gradient_start_allowed)
        self.assertTrue(
            any("prior_ul_mcs_index unique count" in item for item in constant.failures)
        )
        self.assertTrue(
            any("mode 0 prior_ul_mcs_index" in item for item in constant.failures)
        )

        saturated_gate = _gate()
        p40 = next(
            item for item in saturated_gate.state_thresholds if item.name == "scene_p40"
        )
        stricter_p40 = replace(
            p40,
            saturation_lower=0.0,
            saturation_upper=1.0,
            max_boundary_saturation_fraction=0.2,
        )
        saturated_gate = replace(
            saturated_gate,
            state_thresholds=tuple(
                stricter_p40 if item.name == "scene_p40" else item
                for item in saturated_gate.state_thresholds
            ),
        )
        saturated = _filled_ledger(
            saturated_feature="scene_p40", gate=saturated_gate
        ).report()
        self.assertTrue(any("scene_p40 boundary saturation" in item for item in saturated.failures))

        nonfinite = _filled_ledger(nonfinite_feature="scene_si").report()
        self.assertTrue(any("scene_si has" in item and "non-finite" in item for item in nonfinite.failures))

    def test_previous_success_failure_and_metric_variation_are_real_gates(self) -> None:
        all_success = _filled_ledger(all_success=True).report()
        self.assertTrue(any("previous failure count 0" in item for item in all_success.failures))

        constant_quality = _filled_ledger(constant_quality=True).report()
        self.assertTrue(
            any("previous_quality unique count 1" in item for item in constant_quality.failures)
        )

    def test_thresholds_have_no_scientific_defaults_and_enforce_variation(self) -> None:
        signature = inspect.signature(CoverageGateConfig)
        self.assertTrue(
            all(
                parameter.default is inspect.Parameter.empty
                for parameter in signature.parameters.values()
            )
        )
        with self.assertRaises(ExplorationError):
            OutcomeMetricThreshold(
                name="previous_quality",
                min_finite_count=1,
                min_unique_values=1,
                min_span=0.1,
            )


class ActorBehaviorDiagnosticsTests(unittest.TestCase):
    def test_training_is_stochastic_and_evaluation_is_deterministic_only(self) -> None:
        require_selection_path(DecisionPhase.TRAINING, SelectionKind.STOCHASTIC_ACTOR)
        require_selection_path(
            DecisionPhase.EVALUATION, SelectionKind.DETERMINISTIC_ACTOR
        )
        with self.assertRaises(ActorPathError):
            require_selection_path(
                DecisionPhase.TRAINING, SelectionKind.DETERMINISTIC_ACTOR
            )
        with self.assertRaises(ActorPathError):
            require_selection_path(
                DecisionPhase.EVALUATION, SelectionKind.STOCHASTIC_ACTOR
            )

    def test_diagnostics_expose_mode_q_boundary_collapse_and_state_association(self) -> None:
        schedule = _schedule()
        lower, _upper = schedule.config.mode_q_e4_bounds[3]
        rows = []
        for index in range(16):
            observation = CoverageObservation(
                scene_si=float(index),
                scene_p40=float(index) / 20.0,
                prior_ul_mcs_index=index,
                rlc_backlog_bytes=float(index * 1000),
                previous=None,
            )
            rows.append(
                ActionTraceSample(
                    phase=DecisionPhase.TRAINING,
                    selection_kind=SelectionKind.STOCHASTIC_ACTOR,
                    mode_id=3,
                    q_e4=lower,
                    observation=observation,
                )
            )
        diagnostics = summarize_action_trace(rows, schedule)
        self.assertEqual(diagnostics.mode_counts[3], 16)
        self.assertEqual(sum(diagnostics.mode_counts), 16)
        self.assertEqual(diagnostics.dominant_mode_fraction, 1.0)
        self.assertEqual(diagnostics.per_mode_distinct_q_count[3], 1)
        self.assertEqual(diagnostics.per_mode_lower_boundary_count[3], 16)
        self.assertEqual(diagnostics.lower_q_bin_fraction, 1.0)
        self.assertEqual(diagnostics.upper_q_bin_fraction, 0.0)
        self.assertEqual(diagnostics.outer_q_bin_fraction, 1.0)
        self.assertEqual(
            {item.feature_name for item in diagnostics.feature_associations},
            {"scene_si", "scene_p40", "prior_ul_mcs_index", "rlc_backlog_bytes"},
        )
        # Constant q makes the q correlation undefined rather than fabricating zero.
        self.assertTrue(
            all(
                item.q_support_fraction_pearson is None
                for item in diagnostics.feature_associations
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

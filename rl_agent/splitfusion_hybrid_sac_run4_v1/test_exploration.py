from __future__ import annotations

import inspect
import math
import random
import unittest
from dataclasses import fields, replace

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1.exploration import (
    ActionTraceSample,
    ActorPathError,
    CoverageGateConfig,
    CoverageEvidenceClass,
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
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as actions
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    round_half_up_q_e4,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


SESSION = "11111111-1111-4111-8111-111111111111"
UE_ID = "ue-1"
CLOCK = "RUN4_EXPLORATION_TEST_MONOTONIC"
EVIDENCE = "a" * 64


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
    ledger = ExplorationCoverageLedger.for_test_only(schedule, gate or _gate())
    for ordinal, action in enumerate(schedule.actions):
        ledger.record_test_only_decision(
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
    ledger.record_test_only_final_feedback_state(
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


def _metadata(
    identity: contract.DecisionIdentityV1,
    *,
    sample_seq: int,
    kind: contract.MeasurementKind,
    observer: contract.Observer,
    direction: contract.LinkDirection,
    source_ns: int,
    available_ns: int,
) -> contract.MeasurementMetadataV1:
    return contract.MeasurementMetadataV1(
        identity=contract.SampleIdentityV1(
            identity.session_uuid, identity.ue_id, sample_seq
        ),
        kind=kind,
        observer=observer,
        link_direction=direction,
        source="run4-exploration-attested-unit-test",
        source_timestamp_ns=source_ns,
        available_timestamp_ns=available_ns,
        clock_domain=CLOCK,
        valid=True,
    )


def _guarded_state(
    *,
    sequence: int,
    action_open_ns: int,
    previous: contract.PreviousOutcomeV1 | None,
    freshness: contract.FreshnessPolicyV2,
    scaling: contract.EmpiricalScalingV2,
    state_bias: float = 0.0,
) -> tuple[contract.GuardedPolicyStateV2, contract.PolicyFeatureVectorV2]:
    identity = contract.DecisionIdentityV1(SESSION, UE_ID, sequence)
    source_ns = action_open_ns - 30_000_000
    available_ns = action_open_ns - 20_000_000
    common = dict(source_ns=source_ns, available_ns=available_ns)
    scene_identity = 1_000 + sequence
    state = contract.PolicyStateV2(
        identity=identity,
        camera_si=contract.ScalarObservationV1(
            value=10.0 + sequence + state_bias,
            metadata=_metadata(
                identity,
                sample_seq=scene_identity,
                kind=contract.MeasurementKind.CAMERA_SI,
                observer=contract.Observer.SCENE_PIPELINE,
                direction=contract.LinkDirection.NOT_APPLICABLE,
                **common,
            ),
            missing_reason=None,
        ),
        radar_p40=contract.ScalarObservationV1(
            value=0.2 + (sequence % 11) / 20.0,
            metadata=_metadata(
                identity,
                sample_seq=scene_identity,
                kind=contract.MeasurementKind.RADAR_P40,
                observer=contract.Observer.SCENE_PIPELINE,
                direction=contract.LinkDirection.NOT_APPLICABLE,
                **common,
            ),
            missing_reason=None,
        ),
        prior_ul_mcs=contract.PriorUlGrantObservationV1(
            observation=contract.ScalarObservationV1(
                value=sequence % 29,
                metadata=_metadata(
                    identity,
                    sample_seq=2_000 + sequence,
                    kind=(
                        contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX
                    ),
                    observer=contract.Observer.UE,
                    direction=contract.LinkDirection.UPLINK,
                    **common,
                ),
                missing_reason=None,
            ),
            mcs_table=contract.UL_MCS_TABLE_ID,
            harq_round=0,
            new_data_indicator=sequence % 2,
            grant_identity=f"exploration-test-grant-{sequence}",
            scheduler_policy_id=contract.UL_MCS_POLICY_ID,
            selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
        ),
        pre_action_rlc_backlog=contract.ScalarObservationV1(
            value=100 + 37 * sequence,
            metadata=_metadata(
                identity,
                sample_seq=3_000 + sequence,
                kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                observer=contract.Observer.UE,
                direction=contract.LinkDirection.UPLINK,
                **common,
            ),
            missing_reason=None,
        ),
        previous=previous,
    )
    boundary = contract.DecisionBoundaryV1(
        identity=identity,
        state_commit_timestamp_ns=action_open_ns - 10_000_000,
        action_open_timestamp_ns=action_open_ns,
        clock_domain=CLOCK,
    )
    guarded = contract.guard_state_for_action(state, boundary, freshness)
    return guarded, contract.build_policy_features(guarded, scaling)


def _attested_transitions(
    schedule: StratifiedWarmupSchedule,
    action_contract: actions.SplitActionContract,
    *,
    state_bias: float = 0.0,
) -> tuple[contract.SemiMarkovTransitionV2, ...]:
    freshness = contract.FreshnessPolicyV2(
        policy_id="run4-exploration-attested-test",
        policy_version=1,
        evidence_sha256=EVIDENCE,
        camera_si_max_age_ns=50_000_000,
        radar_p40_max_age_ns=50_000_000,
        prior_ul_mcs_max_age_ns=50_000_000,
        pre_action_rlc_backlog_max_age_ns=50_000_000,
    )
    scaling = contract.EmpiricalScalingV2(
        scaling_id="run4-exploration-attested-test",
        scaling_version=1,
        evidence_sha256=EVIDENCE,
        camera_si_center=0.0,
        camera_si_scale=1.0,
        backlog_log1p_scale=10.0,
    )
    current, current_features = _guarded_state(
        sequence=0,
        action_open_ns=2_000_000_000,
        previous=None,
        freshness=freshness,
        scaling=scaling,
        state_bias=state_bias,
    )
    transitions = []
    for ordinal, scheduled in enumerate(schedule.actions):
        executable = action_contract.resolve(
            scheduled.mode_id, scheduled.q_e4 / float(actions.Q_E4_SCALE)
        )
        action = ExecutedActionIdentity.from_executable_action(
            executable, action_contract
        )
        opened = current.boundary.action_open_timestamp_ns
        success = ordinal % 2 == 0
        resolution_offset_ns = 40_000_000 + (ordinal % 100) * 1_000_000
        event = contract.RewardEventV1(
            identity=current.state.identity,
            action=action,
            kind=(
                contract.RewardEventKind.DELIVERED_SUCCESS
                if success
                else contract.RewardEventKind.REGISTERED_SERVICE_FAILURE
            ),
            action_open_timestamp_ns=opened,
            resolution_timestamp_ns=opened + resolution_offset_ns,
            clock_domain=CLOCK,
            source="run4-exploration-attested-unit-test",
            q_perc=(0.5 + (ordinal % 40) / 100.0 if success else None),
        )
        resolution = contract.resolve_reward(event)
        hold = contract.ActionHoldV1(
            identity=current.state.identity,
            action=action,
            tensors=tuple(
                contract.HoldTensorV1(
                    tensor_seq=ordinal * 10 + index,
                    offered_payload_bytes=1_000 + ordinal + index,
                    payload_evidence_class=(
                        contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
                    ),
                    payload_provenance_sha256="b" * 64,
                    reward_requested=index == 0,
                )
                for index in range(2)
            ),
        )
        cycle_end = opened + 200_000_000
        next_state, next_features = _guarded_state(
            sequence=ordinal + 1,
            action_open_ns=cycle_end,
            previous=contract.PreviousOutcomeV1.from_resolution(resolution),
            freshness=freshness,
            scaling=scaling,
            state_bias=state_bias,
        )
        transition = contract.build_transition(
            state=current,
            state_features=current_features,
            action=action,
            hold=hold,
            reward_resolution=resolution,
            next_state=next_state,
            next_state_features=next_features,
            episode_boundary=contract.EpisodeBoundary.CONTINUES,
            duration=2,
            cycle_end_timestamp_ns=cycle_end,
            elapsed_virtual_ns=200_000_000,
            gamma=0.99,
            discount=0.99**2,
        )
        transitions.append(transition)
        current, current_features = next_state, next_features
    return tuple(transitions)


def _filled_attested_ledger(
    action_contract: actions.SplitActionContract,
) -> ExplorationCoverageLedger:
    schedule = _schedule()
    ledger = ExplorationCoverageLedger(schedule, _gate())
    for transition in _attested_transitions(schedule, action_contract):
        ledger.record_transition(transition)
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
    @classmethod
    def setUpClass(cls) -> None:
        cls.action_contract = actions.load_contract()

    def test_complete_real_sequence_passes_all_current_and_previous_strata(self) -> None:
        ledger = _filled_attested_ledger(self.action_contract)
        report = ledger.require_gradient_start()
        self.assertTrue(report.gradient_start_allowed)
        self.assertIs(
            report.evidence_class, CoverageEvidenceClass.TRANSITION_ATTESTED
        )
        self.assertEqual(report.attested_transition_count, len(ledger.schedule))
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
        ledger = ExplorationCoverageLedger.for_test_only(schedule, _gate())
        for ordinal, action in enumerate(schedule.actions[:-1]):
            ledger.record_test_only_decision(
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
        ledger.record_test_only_decision(
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
        ledger = ExplorationCoverageLedger.for_test_only(schedule, _gate())
        with self.assertRaises(CoverageRecordError):
            ledger.record_test_only_decision(
                decision_identity="out-of-order",
                action=schedule.action_at(1),
                observation=_observation(schedule, 0),
            )
        ledger.record_test_only_decision(
            decision_identity="decision-0",
            action=schedule.action_at(0),
            observation=_observation(schedule, 0),
        )
        with self.assertRaises(CoverageRecordError):
            ledger.record_test_only_decision(
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
            ledger.record_test_only_decision(
                decision_identity="decision-0",
                action=schedule.action_at(1),
                observation=_observation(schedule, 1),
            )

    def test_fabricated_bare_variation_cannot_authorize_gradient_start(self) -> None:
        schedule = _schedule()
        production = ExplorationCoverageLedger(schedule, _gate())
        with self.assertRaisesRegex(
            CoverageRecordError, "bare CoverageObservation cannot authorize"
        ):
            production.record_decision(
                decision_identity="fabricated-genesis",
                action=schedule.action_at(0),
                observation=_observation(schedule, 0),
            )

        fabricated = _filled_ledger()
        report = fabricated.report()
        self.assertIs(
            report.evidence_class, CoverageEvidenceClass.TEST_ONLY_UNATTESTED
        )
        self.assertFalse(report.gradient_start_allowed)
        self.assertIn(
            "test-only unattested observations cannot authorize production "
            "gradient start",
            report.failures,
        )
        with self.assertRaises(GradientStartRefused):
            fabricated.require_gradient_start()

    def test_attested_sequence_must_reuse_the_exact_prior_successor(self) -> None:
        schedule = _schedule()
        canonical = _attested_transitions(schedule, self.action_contract)
        independently_redrawn = _attested_transitions(
            schedule, self.action_contract, state_bias=0.125
        )
        ledger = ExplorationCoverageLedger(schedule, _gate())
        ledger.record_transition(canonical[0])
        with self.assertRaisesRegex(CoverageRecordError, "exact prior transition"):
            ledger.record_transition(independently_redrawn[1])

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

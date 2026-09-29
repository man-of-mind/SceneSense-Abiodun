"""Adversarial tests for the event-sourced Run-4 modeled smoke."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_acceptance
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_composite_training as modeled
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as src
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer
from rl_agent.splitfusion_hybrid_sac_run4_v1.test_environment import (
    CLOCK,
    FixtureKernel,
    FixtureProvider,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1.test_modeled_composite_training import (
    ModeledCompositeTrainingTest,
)
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


def _h(char: str) -> str:
    return char * 64


class _AlternatingKernel(FixtureKernel):
    """Synthetic only: alternate valid success and timeout transitions."""

    def __init__(self, decision_ordinal: int) -> None:
        success = decision_ordinal % 2 == 0
        super().__init__(
            kind=(
                contract.RewardEventKind.DELIVERED_SUCCESS
                if success
                else contract.RewardEventKind.TIMEOUT
            ),
            resolution_offset_ns=(
                150_000_000
                if success
                else sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS
            ),
            cycle_offset_ns=200_000_000,
            q_perc=(
                0.55 + 0.01 * (decision_ordinal % 10)
                if success
                else None
            ),
        )

    def configure(self, decision_ordinal: int) -> None:
        success = decision_ordinal % 2 == 0
        self.kind = (
            contract.RewardEventKind.DELIVERED_SUCCESS
            if success
            else contract.RewardEventKind.TIMEOUT
        )
        self.resolution_offset_ns = (
            150_000_000
            if success
            else sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS
        )
        self.q_perc = (
            0.55 + 0.01 * (decision_ordinal % 10) if success else None
        )


class _OrdinalProvider(FixtureProvider):
    """Synthetic provider whose four causal inputs vary by row ordinal."""

    def __init__(self, ordinal: int) -> None:
        super().__init__()
        self.ordinal = ordinal

    def build_state(self, request):
        bundle = super().build_state(request)
        old = bundle.state.state
        state = replace(
            old,
            camera_si=replace(
                old.camera_si, value=0.20 + 0.03 * (self.ordinal % 9)
            ),
            radar_p40=replace(
                old.radar_p40, value=0.15 + 0.04 * (self.ordinal % 8)
            ),
            prior_ul_mcs=replace(
                old.prior_ul_mcs,
                observation=replace(
                    old.prior_ul_mcs.observation, value=4 + (self.ordinal % 20)
                ),
            ),
            pre_action_rlc_backlog=replace(
                old.pre_action_rlc_backlog,
                value=500 + 700 * (self.ordinal % 13),
            ),
        )
        guarded = contract.guard_state_for_action(
            state, bundle.state.boundary, self.freshness
        )
        features = contract.build_policy_features(guarded, self.scaling)
        return environment.DecisionStateBundleV1(guarded, features)


class _FakeTypedCollector:
    """Pure deterministic collector used only to exercise orchestration."""

    SCHEMA = "run4.fake.modeled.collector.v1"

    def __init__(
        self,
        binding: modeled.ModeledCompositeBindingV1,
        mcs_report: dict,
        *,
        constant_core: bool = False,
        wrong_previous_action: bool = False,
    ) -> None:
        self.binding = binding
        self.mcs_report = mcs_report
        self.constant_core = constant_core
        self.wrong_previous_action = wrong_previous_action
        self.catalog = action_contract.load_contract()
        provider = FixtureProvider()
        self.freshness_sha256 = provider.freshness.canonical_sha256()
        self.scaling_sha256 = provider.scaling.canonical_sha256()
        self._history: list[src.CollectedModeledTransitionV1] = []
        self._actions: list[src.ModeledActionRequestV1] = []
        self._pending = None
        self._collector_binding = hashlib.sha256(
            json.dumps(
                {
                    "binding": binding.canonical_sha256,
                    "constant_core": constant_core,
                    "mcs": mcs_report["model_binding_sha256"],
                    "wrong_previous_action": wrong_previous_action,
                    "schema": self.SCHEMA,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

    @property
    def collector_binding_sha256(self) -> str:
        return self._collector_binding

    @property
    def decision_count(self) -> int:
        return len(self._history)

    def _action(self, mode_id: int, q_e4: int) -> ExecutedActionIdentity:
        executable = self.catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, self.catalog
        )

    @staticmethod
    def _session(ordinal: int) -> str:
        return f"00000000-0000-4000-8000-{ordinal + 1:012d}"

    def _prepare(self) -> None:
        if self._pending is not None:
            return
        ordinal = self.decision_count
        provider = _OrdinalProvider(0 if self.constant_core else ordinal)
        kernel = _AlternatingKernel(max(0, ordinal - 1))
        calibration = environment.CalibrationBindingV1(
            calibration_id="test-only-modeled-smoke-collector",
            calibration_version=1,
            evidence_sha256=_h("1"),
            verifier_report_sha256=_h("2"),
            kernel_binding_sha256=_h("3"),
            state_provider_binding_sha256=_h("4"),
        )
        with mock.patch.object(
            environment,
            "REGISTERED_CALIBRATION_BINDING_SHA256",
            calibration.canonical_sha256(),
        ):
            env = environment.Run4SequentialEnvironmentV1(
                state_provider=provider,
                kernel=kernel,
                gamma=0.99,
                evidence_class=(
                    environment.EnvironmentEvidenceClass.CALIBRATED_EMPIRICAL
                ),
                calibration_binding=calibration,
            )
        env.reset(session_uuid=self._session(ordinal), ue_id="ue-test")
        # Every non-genesis row is primed by the actual preceding action and
        # outcome. This is a causal chain, not decorative previous-state data.
        if ordinal > 0:
            previous = self._actions[-1]
            primer_mode = previous.mode_id
            primer_q = previous.q_e4
            if self.wrong_previous_action:
                primer_mode = (primer_mode + 1) % 12
            env.step(self._action(primer_mode, primer_q))
            kernel.configure(ordinal)
        self._pending = (env, kernel)

    def current_state_features(self) -> tuple[float, ...]:
        self._prepare()
        env, _ = self._pending
        return tuple(float(item) for item in env.current_state.features.as_tuple())

    def _support(self, mode_id: int) -> modeled.ModeledCompositeSupportUseV1:
        return modeled.ModeledCompositeSupportUseV1(
            target_profile_label="FAVORABLE_STABLE",
            target_mode_id=mode_id,
            source_profile_labels=("FAVORABLE_STABLE",),
            source_mode_ids=tuple(range(12)),
            profile_transfer_status=(
                modeled.ProfileTransferStatus.PROFILE_WITHIN_DIRECT_FIT_SUPPORT
            ),
            mode_transfer_status=(
                modeled.ModeTransferStatus.MODE_WITHIN_DIRECT_FIT_SUPPORT
            ),
            payload_in_fit_support=True,
            backlog_in_fit_support=True,
            mcs_in_fit_support=True,
            quality_in_fit_support=True,
            total_latency_residual_in_fit_support=True,
            widened_uncertainty_applied=False,
            support_evidence_sha256=_h("9"),
        )

    @staticmethod
    def _endpoints(
        action_open_ns: int, total_ns: int
    ) -> modeled.FeedbackEndpointPairV1:
        actor_ns = 750_000
        dispatch_ns = 250_000
        return modeled.FeedbackEndpointPairV1(
            action_open_timestamp_ns=action_open_ns,
            feedback_received_timestamp_ns=action_open_ns + total_ns,
            clock_domain=CLOCK,
            source_row_sha256=_h("8"),
            fixed_action_source_total_ns=total_ns - actor_ns - dispatch_ns,
            fixed_action_source_total_evidence_sha256=_h("7"),
            actor_inference_ns=actor_ns,
            actor_inference_evidence_sha256=_h("6"),
            quantization_dispatch_ns=dispatch_ns,
            quantization_dispatch_evidence_sha256=_h("5"),
        )

    def collect(
        self, request: src.ModeledActionRequestV1
    ) -> src.CollectedModeledTransitionV1:
        if request.decision_ordinal != self.decision_count:
            raise RuntimeError("fake collector received an out-of-order action")
        before = self.current_state_features()
        env, kernel = self._pending
        cycle = env.step(self._action(request.mode_id, request.q_e4))
        transition = cycle.export_for_replay()
        success = kernel.kind is contract.RewardEventKind.DELIVERED_SUCCESS
        actual_total = 150_000_000 if success else 220_000_000
        projection = modeled.LatencyProjectionV1.from_ordered_endpoints(
            self._endpoints(
                transition.state.boundary.action_open_timestamp_ns,
                actual_total,
            )
        )
        envelope = modeled.ModeledCompositeTrainingIssuerV1(
            self.binding
        ).issue(
            transition=transition,
            support_use=self._support(request.mode_id),
            latency_projection=projection,
        )
        wrapper = envelope.export_for_offline_training()
        record = src.CollectedModeledTransitionV1(
            request=request,
            wrapper=wrapper,
            state_features=before,
            state_features_sha256=hashlib.sha256(
                json.dumps(
                    list(before),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
            duration=transition.duration,
            terminal=wrapper.terminal,
            reward=wrapper.reward,
            q_perc=wrapper.q_perc,
            latency_ms=wrapper.latency_ms,
            modeled_binding_sha256=self.binding.canonical_sha256,
            mcs_acceptance_result_sha256=(
                mcs_transition_acceptance
                .REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256
            ),
            mcs_model_binding_sha256=self.mcs_report["model_binding_sha256"],
            source_partition=src.FIT_PARTITION_LABEL,
            validation_evidence_consumed=False,
        )
        self._history.append(record)
        self._actions.append(request)
        self._pending = None
        return record

    def checkpoint(self) -> src.CollectorCheckpointV1:
        payload = json.dumps(
            {
                "actions": [item.to_dict() for item in self._actions],
                "schema": self.SCHEMA,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return src.CollectorCheckpointV1(
            collector_schema_id=self.SCHEMA,
            collector_binding_sha256=self.collector_binding_sha256,
            decision_count=self.decision_count,
            transition_sha256s=tuple(
                item.wrapper.transition_sha256 for item in self._history
            ),
            payload_json=payload,
            payload_sha256=hashlib.sha256(payload.encode("ascii")).hexdigest(),
        )

    def restore(self, checkpoint: src.CollectorCheckpointV1) -> None:
        if checkpoint.collector_binding_sha256 != self.collector_binding_sha256:
            raise RuntimeError("fake collector binding mismatch")
        raw = json.loads(checkpoint.payload_json)
        self._history.clear()
        self._actions.clear()
        self._pending = None
        for item in raw["actions"]:
            request = src.ModeledActionRequestV1(
                decision_ordinal=item["decision_ordinal"],
                mode_id=item["mode_id"],
                q_e4=item["q_e4"],
                source=item["source"],
                warmup_q_bin_index=item["warmup_q_bin_index"],
            )
            self.collect(request)
        if tuple(item.wrapper.transition_sha256 for item in self._history) != (
            checkpoint.transition_sha256s
        ):
            raise RuntimeError("fake collector replay changed")

    def history(self) -> tuple[src.CollectedModeledTransitionV1, ...]:
        return tuple(self._history)


class ModeledSmokeOrchestratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        ModeledCompositeTrainingTest.setUpClass()
        fixture = ModeledCompositeTrainingTest(
            "test_binding_is_explicitly_offline_and_not_empirical"
        )
        base = fixture.binding()
        cls.mcs_report = mcs_transition_acceptance.load_registered_acceptance()
        disclosures = []
        for item in base.component_disclosures:
            if item.role is modeled.ComponentRole.UL_MCS_TRANSITION:
                item = replace(
                    item,
                    source_evidence_sha256=cls.mcs_report[
                        "source_evidence_sha256"
                    ],
                    fit_support_sha256=cls.mcs_report["model_binding_sha256"],
                )
            disclosures.append(item)
        cls.binding = replace(base, component_disclosures=tuple(disclosures))
        cls.variation_contract = src.PreflightVariationContractV1(
            contract_id="synthetic-test-preflight-variation",
            contract_version=1,
            evidence_sha256=_h("a"),
            feature_schema_sha256=contract.FEATURE_SCHEMA_SHA256,
            source_partition=src.FIT_PARTITION_LABEL,
            requirements=(
                src.PreflightFeatureRequirementV1(
                    "camera_si_scaled", 9, 0.03 * 8
                ),
                src.PreflightFeatureRequirementV1(
                    "radar_p40", 8, 0.04 * 7
                ),
                src.PreflightFeatureRequirementV1(
                    "prior_ul_mcs_normalized", 20, 19.0 / 28.0
                ),
                src.PreflightFeatureRequirementV1(
                    "pre_action_rlc_backlog_log1p_scaled",
                    13,
                    (
                        math.log1p(8900) / 10.0
                        - math.log1p(500) / 10.0
                    ),
                ),
            ),
        )
        probe = _FakeTypedCollector(cls.binding, cls.mcs_report)
        cls.factory = src.ModeledSmokeRunnerFactoryV1(
            modeled_binding=cls.binding,
            gamma=0.99,
            freshness_policy_sha256=probe.freshness_sha256,
            empirical_scaling_sha256=probe.scaling_sha256,
            trainer_config=trainer.TrainerConfigV1(
                alpha_d=smoke_preregistration.FROZEN_CONFIG.alpha_d,
                alpha_c=smoke_preregistration.FROZEN_CONFIG.alpha_c,
                tau=smoke_preregistration.FROZEN_CONFIG.polyak_tau,
                actor_lr=(
                    smoke_preregistration.FROZEN_CONFIG.actor_learning_rate
                ),
                critic_lr=(
                    smoke_preregistration.FROZEN_CONFIG.critic_learning_rate
                ),
                nominal_batch_size=(
                    smoke_preregistration.FROZEN_CONFIG.batch_size
                ),
            ),
            seed_plan=src.RunnerSeedPlanV1.seed17(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def collector_factory(self):
        return _FakeTypedCollector(self.binding, self.mcs_report)

    def orchestrator(self) -> src.ModeledSmokeOrchestratorV1:
        return src.ModeledSmokeOrchestratorV1(
            runner_factory=self.factory,
            collector_factory=self.collector_factory,
            preflight_variation_contract=self.variation_contract,
        )

    def test_frozen_warmup_is_exact_balanced_288(self) -> None:
        schedules = []
        plans = []
        for seed in smoke_preregistration.FROZEN_CONFIG.seed_order:
            plan = src.RunnerSeedPlanV1.for_registered_seed(seed)
            schedule = src.build_frozen_warmup_schedule(seed)
            plans.append(plan)
            schedules.append(schedule)
            self.assertEqual(len(schedule), 288)
            counts = {}
            for item in schedule.actions:
                counts[(item.mode_id, item.q_bin_index)] = (
                    counts.get((item.mode_id, item.q_bin_index), 0) + 1
                )
            self.assertEqual(set(counts.values()), {4})
            self.assertEqual(len(counts), 72)
        self.assertEqual(len({item.canonical_sha256 for item in plans}), 3)
        self.assertEqual(
            len({item.config.schedule_id for item in schedules}), 3
        )
        with self.assertRaises(src.ModeledSmokeBindingError):
            src.RunnerSeedPlanV1.for_registered_seed(99)
        with self.assertRaises(src.ModeledSmokeScheduleError):
            src.build_frozen_warmup_schedule(99)

    def test_no_gradient_preflight_is_exact_and_auditable(self) -> None:
        orchestrator = self.orchestrator()
        global_before = torch.random.get_rng_state().clone()
        report = orchestrator.run_no_gradient_preflight()
        self.assertEqual(report.decision_count, 288)
        self.assertEqual(report.success_count, 144)
        self.assertEqual(report.failure_count, 144)
        self.assertEqual(len(report.feature_diagnostics), 21)
        self.assertTrue(
            all(item.finite_count == 288 for item in report.feature_diagnostics)
        )
        self.assertEqual(
            report.previous_reconciliation.previous_success_count, 144
        )
        self.assertEqual(
            report.previous_reconciliation.previous_failure_count, 143
        )
        self.assertTrue(report.gradient_free)
        self.assertEqual(orchestrator.update_count, 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        checkpoint = orchestrator.checkpoint()
        payload = src.checkpoint_to_bytes(checkpoint)
        self.assertEqual(payload, src.checkpoint_to_bytes(checkpoint))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "checkpoint_000.json"
            file_sha = src.write_checkpoint(path, checkpoint)
            self.assertEqual(file_sha, hashlib.sha256(payload).hexdigest())
            decoded = src.read_checkpoint(path)
            self.assertEqual(decoded.canonical_sha256, checkpoint.canonical_sha256)
            with self.assertRaises(src.ModeledSmokeCheckpointError):
                src.write_checkpoint(path, checkpoint)
        self.assertEqual(
            src.checkpoint_from_bytes(payload).canonical_sha256,
            checkpoint.canonical_sha256,
        )
        damaged = payload.replace(b'"decision_count":288', b'"decision_count":289', 1)
        with self.assertRaises(src.ModeledSmokeCheckpointError):
            src.checkpoint_from_bytes(damaged)

    def test_wrong_mcs_binding_is_refused_before_replay_mutation(self) -> None:
        orchestrator = self.orchestrator()
        request = orchestrator._warmup_request(0)
        record = orchestrator.collector.collect(request)
        forged = replace(record, mcs_model_binding_sha256="0" * 64)
        with self.assertRaisesRegex(src.ModeledSmokeBindingError, "MCS model"):
            orchestrator._validate_collected(forged, request)
        self.assertEqual(orchestrator.runner.replay_buffer.accepted_count, 0)
        self.assertEqual(orchestrator.decision_count, 0)

    def test_validation_evidence_is_refused(self) -> None:
        orchestrator = self.orchestrator()
        request = orchestrator._warmup_request(0)
        record = orchestrator.collector.collect(request)
        with self.assertRaises(src.ModeledSmokeBindingError):
            replace(record, validation_evidence_consumed=True)
        self.assertEqual(orchestrator.runner.replay_buffer.accepted_count, 0)

    def test_constant_core_features_fail_even_when_previous_state_varies(self) -> None:
        collector_factory = lambda: _FakeTypedCollector(
            self.binding, self.mcs_report, constant_core=True
        )
        orchestrator = src.ModeledSmokeOrchestratorV1(
            runner_factory=self.factory,
            collector_factory=collector_factory,
            preflight_variation_contract=self.variation_contract,
        )
        with self.assertRaisesRegex(
            src.ModeledSmokePreflightError, "camera_si_scaled"
        ):
            orchestrator.run_no_gradient_preflight()
        # This is the exact old false-pass shape: state varies through previous
        # fields, yet the four causal scene/radio inputs are constant.
        self.assertGreater(len({item.state_features for item in orchestrator._history}), 1)
        self.assertEqual(orchestrator.runner.replay_buffer.accepted_count, 288)
        self.assertEqual(orchestrator.update_count, 0)

    def test_wrong_previous_action_is_refused_before_second_replay_insert(self) -> None:
        collector_factory = lambda: _FakeTypedCollector(
            self.binding, self.mcs_report, wrong_previous_action=True
        )
        orchestrator = src.ModeledSmokeOrchestratorV1(
            runner_factory=self.factory,
            collector_factory=collector_factory,
            preflight_variation_contract=self.variation_contract,
        )
        orchestrator.collect_one()
        with self.assertRaisesRegex(
            src.ModeledSmokeBindingError, "previous action differs"
        ):
            orchestrator.collect_one()
        self.assertEqual(orchestrator.runner.replay_buffer.accepted_count, 1)
        self.assertEqual(orchestrator.decision_count, 1)

    def test_resume_from_250_is_bit_identical_at_500(self) -> None:
        original = self.orchestrator()
        events = {}

        def capture(event: src.ModeledSmokeCheckpointEventV1) -> None:
            events[event.update] = event.checkpoint

        summary = original.run_to_hard_stop(checkpoint_callback=capture)
        self.assertEqual(summary.final_update, 500)
        self.assertEqual(summary.final_decision_count, 2288)
        self.assertEqual(tuple(events), (0, 100, 250, 500))
        restored = src.ModeledSmokeOrchestratorV1.restore(
            events[250],
            runner_factory=self.factory,
            collector_factory=self.collector_factory,
            preflight_variation_contract=self.variation_contract,
        )
        resumed_events = {}
        resumed = restored.run_to_hard_stop(
            checkpoint_callback=lambda event: resumed_events.setdefault(
                event.update, event.checkpoint
            )
        )
        self.assertEqual(resumed.starting_update, 250)
        self.assertEqual(tuple(resumed_events), (500,))
        self.assertEqual(
            resumed_events[500].canonical_sha256,
            events[500].canonical_sha256,
        )
        self.assertEqual(resumed_events[500].boundary, events[500].boundary)


if __name__ == "__main__":
    unittest.main()

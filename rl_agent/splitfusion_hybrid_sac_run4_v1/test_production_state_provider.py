"""Adversarial tests for the pure Run-4 production-state provider."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from dataclasses import replace

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import fit_scene_provider
from rl_agent.splitfusion_hybrid_sac_run4_v1 import production_state_provider as src
from rl_agent.splitfusion_hybrid_sac_run4_v1 import quality_adapter
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


SESSION = "99999999-9999-4999-8999-999999999999"
UE_ID = "ue-run4-state-provider"
CLOCK = "RUN4_TEST_MONOTONIC"
FIT_BINDING = "1" * 64
RADIO_BINDING = "2" * 64
CALIBRATION = "3" * 64
REPORT = "4" * 64
MCS_PROVENANCE = "5" * 64
BACKLOG_PROVENANCE = "6" * 64
SURFACE_BINDING = "7" * 64


class ProductionStateProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.action_catalog = action_contract.load_contract()

    def action(self, mode_id: int = 4, q_e4: int = 7000) -> ExecutedActionIdentity:
        executable = self.action_catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, self.action_catalog
        )

    def prerequisites(
        self,
        *,
        status: str = src.TEST_ONLY_VERIFICATION_STATUS,
        max_age_ns: int = 100_000_000,
    ) -> src.StateProviderPrerequisitesV1:
        return src.StateProviderPrerequisitesV1(
            fit_scene_provider_binding_sha256=FIT_BINDING,
            radio_queue_evidence_sha256=RADIO_BINDING,
            calibration_evidence_sha256=CALIBRATION,
            verifier_report_sha256=REPORT,
            scaling=contract.EmpiricalScalingV2(
                scaling_id="test-only-run4-state-scaling",
                scaling_version=1,
                evidence_sha256=CALIBRATION,
                camera_si_center=10.0,
                camera_si_scale=5.0,
                backlog_log1p_scale=math.log1p(1023),
            ),
            freshness=contract.FreshnessPolicyV2(
                policy_id="test-only-run4-state-freshness",
                policy_version=1,
                evidence_sha256=CALIBRATION,
                camera_si_max_age_ns=max_age_ns,
                radar_p40_max_age_ns=max_age_ns,
                prior_ul_mcs_max_age_ns=max_age_ns,
                pre_action_rlc_backlog_max_age_ns=max_age_ns,
            ),
            verification_status=status,
        )

    def provider(self) -> src.Run4ProductionStateProviderV1:
        prerequisites = self.prerequisites()
        return src.Run4ProductionStateProviderV1(
            prerequisites=prerequisites,
            authorization=src.authorize_test_only_state_provider(prerequisites),
        )

    def identity(self, sequence: int) -> contract.DecisionIdentityV1:
        return contract.DecisionIdentityV1(SESSION, UE_ID, sequence)

    def boundary(
        self,
        sequence: int,
        *,
        commit_ns: int = 1_000_000_000,
        open_ns: int = 1_010_000_000,
    ) -> contract.DecisionBoundaryV1:
        return contract.DecisionBoundaryV1(
            identity=self.identity(sequence),
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=open_ns,
            clock_domain=CLOCK,
        )

    def scene_draw(
        self,
        *,
        camera_si: float = 20.0,
        radar_p40: float = 0.4,
        ordinal: int = 1,
        provider_binding: str = FIT_BINDING,
    ) -> fit_scene_provider.FitSceneDrawV1:
        selection = quality_adapter.FitSceneSelectionV1(
            sample_id=f"hidden-sample-{ordinal}",
            episode_id="hidden-fit-episode",
            frame_id=1000 + ordinal,
            grid_split="fit",
            selection_rank_within_fit=ordinal,
            inclusion_probability=0.5,
            sampling_weight=2.0,
            camera_si=camera_si,
            radar_p40=radar_p40,
            surface_binding_sha256=SURFACE_BINDING,
        )
        return fit_scene_provider.FitSceneDrawV1(
            provider_binding_sha256=provider_binding,
            draw_ordinal=ordinal,
            rng_draw=0.25,
            selection=selection,
        )

    def radio_state(
        self,
        sequence: int,
        *,
        mcs: int | None = 12,
        backlog: int | None = 1023,
        mcs_provenance: str = MCS_PROVENANCE,
        backlog_provenance: str = BACKLOG_PROVENANCE,
    ) -> sequential_kernel.RadioQueueStateV1:
        source_seq = max(0, sequence - 1)
        return sequential_kernel.RadioQueueStateV1(
            session_uuid=SESSION,
            ue_id=UE_ID,
            decision_seq=sequence,
            prior_ul_mcs=sequential_kernel.IntegerObservationV1(
                value=mcs,
                missing_reason=None if mcs is not None else "missing MCS",
                source_decision_seq=source_seq,
                provenance_sha256=mcs_provenance,
            ),
            pre_enqueue_backlog_bytes=sequential_kernel.IntegerObservationV1(
                value=backlog,
                missing_reason=None if backlog is not None else "missing backlog",
                source_decision_seq=source_seq,
                provenance_sha256=backlog_provenance,
            ),
        )

    def timing(
        self,
        source: str,
        *,
        source_ns: int = 950_000_000,
        available_ns: int = 980_000_000,
        clock: str = CLOCK,
    ) -> src.ObservationTimingV1:
        return src.ObservationTimingV1(
            source=source,
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=clock,
        )

    def previous(
        self,
        *,
        mode_id: int = 4,
        q_e4: int = 7000,
        q_perc: float = 0.8,
        latency_ms: float = 120.0,
        terminal: contract.RewardTerminal = contract.RewardTerminal.SUCCESS,
    ) -> contract.PreviousOutcomeV1:
        success = terminal is contract.RewardTerminal.SUCCESS
        return contract.PreviousOutcomeV1(
            identity=self.identity(0),
            action=self.action(mode_id, q_e4),
            terminal=terminal,
            q_perc=q_perc if success else None,
            latency_ms=latency_ms if success else None,
            available_timestamp_ns=1_100_000_000,
            clock_domain=CLOCK,
            reward_resolution_sha256="8" * 64,
        )

    def staged(
        self,
        sequence: int,
        *,
        previous: contract.PreviousOutcomeV1 | None = None,
        draw: fit_scene_provider.FitSceneDrawV1 | None = None,
        radio: sequential_kernel.RadioQueueStateV1 | None = None,
        boundary: contract.DecisionBoundaryV1 | None = None,
        scene_timing: src.ObservationTimingV1 | None = None,
        mcs_timing: src.ObservationTimingV1 | None = None,
        backlog_timing: src.ObservationTimingV1 | None = None,
        mcs_provenance: str = MCS_PROVENANCE,
        backlog_provenance: str = BACKLOG_PROVENANCE,
    ) -> src.StagedDecisionInputsV1:
        if boundary is None:
            offset = sequence * 200_000_000
            boundary = self.boundary(
                sequence,
                commit_ns=1_000_000_000 + offset,
                open_ns=1_010_000_000 + offset,
            )
        default_source = boundary.state_commit_timestamp_ns - 50_000_000
        default_available = boundary.state_commit_timestamp_ns - 20_000_000
        if radio is None:
            radio = self.radio_state(sequence)
        return src.StagedDecisionInputsV1(
            identity=self.identity(sequence),
            boundary=boundary,
            scene_draw=(
                self.scene_draw(ordinal=sequence + 1) if draw is None else draw
            ),
            scene_timing=(
                self.timing(
                    "fit-scene-pipeline",
                    source_ns=default_source,
                    available_ns=default_available,
                )
                if scene_timing is None
                else scene_timing
            ),
            radio_state=radio,
            prior_grant=src.PriorGrantProvenanceV1(
                radio_observation_provenance_sha256=mcs_provenance,
                grant_identity=f"ue-ul-dci:{sequence}",
                new_data_indicator=sequence % 2,
                harq_round=0,
                mcs_table=contract.UL_MCS_TABLE_ID,
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                timing=(
                    self.timing(
                        "ue-decoded-ul-dci",
                        source_ns=default_source,
                        available_ns=default_available,
                    )
                    if mcs_timing is None
                    else mcs_timing
                ),
            ),
            backlog=src.BacklogProvenanceV1(
                radio_observation_provenance_sha256=backlog_provenance,
                timing=(
                    self.timing(
                        "ue-pre-enqueue-rlc",
                        source_ns=default_source,
                        available_ns=default_available,
                    )
                    if backlog_timing is None
                    else backlog_timing
                ),
            ),
            expected_previous_sha256=(
                None if previous is None else previous.canonical_sha256()
            ),
        )

    def request(
        self,
        sequence: int,
        *,
        previous: contract.PreviousOutcomeV1 | None = None,
        required_open_ns: int | None = None,
        minimum_commit_ns: int = 0,
    ) -> environment.DecisionStateRequestV1:
        return environment.DecisionStateRequestV1(
            identity=self.identity(sequence),
            previous=previous,
            required_action_open_timestamp_ns=required_open_ns,
            minimum_state_commit_timestamp_ns=minimum_commit_ns,
        )

    def build(
        self,
        provider: src.Run4ProductionStateProviderV1,
        staged: src.StagedDecisionInputsV1,
        previous: contract.PreviousOutcomeV1 | None = None,
    ) -> environment.DecisionStateBundleV1:
        provider.stage_decision(staged)
        return provider.build_state(
            self.request(
                staged.identity.decision_seq,
                previous=previous,
                required_open_ns=(
                    None
                    if staged.identity.decision_seq == 0
                    else staged.boundary.action_open_timestamp_ns
                ),
                minimum_commit_ns=(
                    0 if previous is None else previous.available_timestamp_ns
                ),
            )
        )

    def test_genesis_all_21_features_and_genuine_zero_are_preserved(self) -> None:
        provider = self.provider()
        radio = self.radio_state(0, mcs=0, backlog=0)
        draw = self.scene_draw(camera_si=20.0, radar_p40=0.4)
        bundle = self.build(
            provider, self.staged(0, draw=draw, radio=radio)
        )
        values = bundle.features.as_dict()
        self.assertEqual(tuple(values), contract.POLICY_FEATURE_ORDER)
        self.assertEqual(len(values), 21)
        self.assertEqual(values["camera_si_scaled"], 2.0)
        self.assertEqual(values["radar_p40"], 0.4)
        self.assertEqual(values["prior_ul_mcs_normalized"], 0.0)
        self.assertEqual(values["pre_action_rlc_backlog_log1p_scaled"], 0.0)
        self.assertEqual(bundle.state.state.prior_ul_mcs.observation.value, 0)
        self.assertEqual(bundle.state.state.pre_action_rlc_backlog.value, 0)
        self.assertEqual(tuple(values.values())[4:], (0.0,) * 17)

        # Hidden frame/session identifiers remain metadata; actor output is
        # exactly the frozen float-only allow-list.
        feature_json = json.dumps(bundle.features.to_canonical_dict())
        self.assertNotIn(draw.selection.sample_id, feature_json)
        self.assertNotIn(draw.selection.episode_id, feature_json)
        self.assertNotIn(SESSION, feature_json)
        self.assertTrue(all(type(value) is float for value in values.values()))

    def test_current_scene_radio_fixed_previous_action_outcome_changes_tail(self) -> None:
        first = self.previous(mode_id=2, q_e4=3000, q_perc=0.65, latency_ms=80.0)
        second = self.previous(mode_id=9, q_e4=9000, q_perc=0.90, latency_ms=160.0)

        def vector(previous: contract.PreviousOutcomeV1) -> tuple[float, ...]:
            provider = self.provider()
            staged = self.staged(
                1,
                previous=previous,
                draw=self.scene_draw(
                    camera_si=20.0, radar_p40=0.4, ordinal=2
                ),
                radio=self.radio_state(1, mcs=12, backlog=1023),
            )
            # A continuation-only provider must first consume genesis.
            self.build(provider, self.staged(0))
            return self.build(provider, staged, previous).features.as_tuple()

        left = vector(first)
        right = vector(second)
        self.assertEqual(left[:4], right[:4])
        self.assertNotEqual(left[4:], right[4:])
        self.assertEqual(left[4 + 2], 1.0)
        self.assertEqual(right[4 + 9], 1.0)
        self.assertEqual(
            left[-5:], (3000.0 / 9800.0, 0.65, 80.0 / 170.0, 1.0, 1.0)
        )
        self.assertEqual(
            right[-5:], (9000.0 / 9800.0, 0.9, 160.0 / 170.0, 1.0, 1.0)
        )

    def test_previous_fixed_current_scene_radio_changes_only_first_four(self) -> None:
        previous = self.previous()

        def vector(
            camera: float, radar: float, mcs: int, backlog: int
        ) -> tuple[float, ...]:
            provider = self.provider()
            self.build(provider, self.staged(0))
            staged = self.staged(
                1,
                previous=previous,
                draw=self.scene_draw(
                    camera_si=camera, radar_p40=radar, ordinal=2
                ),
                radio=self.radio_state(1, mcs=mcs, backlog=backlog),
            )
            return self.build(provider, staged, previous).features.as_tuple()

        left = vector(15.0, 0.2, 5, 10)
        right = vector(30.0, 0.8, 25, 5000)
        self.assertNotEqual(left[:4], right[:4])
        self.assertEqual(left[4:], right[4:])

    def test_failure_previous_is_present_without_fabricated_quality_latency(self) -> None:
        previous = self.previous(
            terminal=contract.RewardTerminal.TIMEOUT,
        )
        provider = self.provider()
        self.build(provider, self.staged(0))
        bundle = self.build(
            provider, self.staged(1, previous=previous), previous
        )
        values = bundle.features.as_dict()
        self.assertEqual(values[f"prev_joint_mode_{previous.action.mode_id}_one_hot"], 1.0)
        self.assertEqual(values["prev_q_normalized"], 7000.0 / 9800.0)
        self.assertEqual(values["prev_quality_qperc"], 0.0)
        self.assertEqual(values["prev_latency_normalized"], 0.0)
        self.assertEqual(values["prev_present"], 1.0)
        self.assertEqual(values["prev_success"], 0.0)

    def test_missing_radio_never_becomes_numeric_zero(self) -> None:
        provider = self.provider()
        missing_mcs = self.radio_state(0, mcs=None)
        with self.assertRaisesRegex(
            src.StateAssemblyFallbackRequired, "zero-fill forbidden"
        ):
            provider.stage_decision(self.staged(0, radio=missing_mcs))

        missing_backlog = self.radio_state(0, backlog=None)
        with self.assertRaisesRegex(
            src.StateAssemblyFallbackRequired, "zero-fill forbidden"
        ):
            provider.stage_decision(self.staged(0, radio=missing_backlog))

    def test_radio_values_are_not_forward_filled_after_a_valid_decision(self) -> None:
        provider = self.provider()
        self.build(provider, self.staged(0, radio=self.radio_state(0, mcs=21)))
        previous = self.previous()
        missing_successor = self.radio_state(1, mcs=None)
        with self.assertRaisesRegex(
            src.StateAssemblyFallbackRequired, "zero-fill forbidden"
        ):
            provider.stage_decision(
                self.staged(
                    1,
                    previous=previous,
                    radio=missing_successor,
                )
            )

    def test_radio_identity_and_timestamp_order_are_not_inferred(self) -> None:
        foreign_radio = replace(self.radio_state(0), ue_id="foreign-ue")
        with self.assertRaisesRegex(
            src.StateProviderEvidenceError, "exact decision identity"
        ):
            self.staged(0, radio=foreign_radio)
        with self.assertRaisesRegex(
            src.StateProviderEvidenceError, "source_timestamp_ns"
        ):
            self.timing(
                "reversed-timing",
                source_ns=981_000_000,
                available_ns=980_000_000,
            )

    def test_mismatched_provenance_envelopes_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            src.StateProviderEvidenceError, "prior-grant envelope"
        ):
            self.staged(0, mcs_provenance="a" * 64)
        with self.assertRaisesRegex(
            src.StateProviderEvidenceError, "backlog envelope"
        ):
            self.staged(0, backlog_provenance="b" * 64)

    def test_foreign_scene_provider_binding_is_rejected(self) -> None:
        provider = self.provider()
        staged = self.staged(
            0, draw=self.scene_draw(provider_binding="f" * 64)
        )
        with self.assertRaisesRegex(
            src.StateProviderEvidenceError, "fit-scene draw/provider"
        ):
            provider.stage_decision(staged)

    def test_stale_late_and_foreign_clock_inputs_fail_closed(self) -> None:
        stale_provider = src.Run4ProductionStateProviderV1(
            prerequisites=self.prerequisites(max_age_ns=10_000_000),
            authorization=src.authorize_test_only_state_provider(
                self.prerequisites(max_age_ns=10_000_000)
            ),
        )
        stale = self.staged(0)
        stale_provider.stage_decision(stale)
        with self.assertRaisesRegex(src.StateAssemblyFallbackRequired, "stale"):
            stale_provider.build_state(self.request(0))

        late = self.staged(
            0,
            mcs_timing=self.timing(
                "late-ul-dci",
                source_ns=990_000_000,
                available_ns=1_001_000_000,
            ),
        )
        provider = self.provider()
        provider.stage_decision(late)
        with self.assertRaisesRegex(
            src.StateAssemblyFallbackRequired, "not available"
        ):
            provider.build_state(self.request(0))

        with self.assertRaisesRegex(
            src.StateProviderEvidenceError, "different clock domain"
        ):
            self.staged(
                0,
                backlog_timing=self.timing(
                    "foreign-clock-backlog", clock="FOREIGN_CLOCK"
                ),
            )

    def test_exact_previous_digest_and_environment_timestamps_are_enforced(self) -> None:
        previous = self.previous()
        provider = self.provider()
        self.build(provider, self.staged(0))
        staged = self.staged(1, previous=previous)
        provider.stage_decision(staged)

        other = replace(previous, reward_resolution_sha256="9" * 64)
        with self.assertRaisesRegex(src.StateSequenceError, "exact staged previous"):
            provider.build_state(
                self.request(
                    1,
                    previous=other,
                    required_open_ns=staged.boundary.action_open_timestamp_ns,
                    minimum_commit_ns=previous.available_timestamp_ns,
                )
            )
        # Failure is atomic: the exact request can still consume the stage.
        bundle = provider.build_state(
            self.request(
                1,
                previous=previous,
                required_open_ns=staged.boundary.action_open_timestamp_ns,
                minimum_commit_ns=previous.available_timestamp_ns,
            )
        )
        self.assertEqual(
            bundle.state.state.previous.canonical_sha256(),
            previous.canonical_sha256(),
        )

        fresh = self.provider()
        next_staged = self.staged(0)
        fresh.stage_decision(next_staged)
        with self.assertRaisesRegex(src.StateSequenceError, "action-open"):
            fresh.build_state(
                self.request(0, required_open_ns=123)
            )

    def test_non_genesis_cannot_reset_skip_or_start_without_genesis(self) -> None:
        provider = self.provider()
        previous = self.previous()
        with self.assertRaisesRegex(src.StateSequenceError, "start at genesis"):
            provider.stage_decision(self.staged(1, previous=previous))

        self.build(provider, self.staged(0))
        with self.assertRaisesRegex(src.StateSequenceError, "cannot reset or skip"):
            provider.stage_decision(self.staged(0))

        skipped_previous = replace(
            previous,
            identity=contract.DecisionIdentityV1(SESSION, UE_ID, 1),
        )
        with self.assertRaisesRegex(src.StateSequenceError, "cannot reset or skip"):
            provider.stage_decision(self.staged(2, previous=skipped_previous))

    def test_pending_stage_cannot_be_overwritten_or_reused(self) -> None:
        provider = self.provider()
        staged = self.staged(0)
        provider.stage_decision(staged)
        with self.assertRaisesRegex(src.StateSequenceError, "cannot be overwritten"):
            provider.stage_decision(staged)
        provider.build_state(self.request(0))
        with self.assertRaisesRegex(src.StateSequenceError, "no causal"):
            provider.build_state(self.request(0))

    def test_test_authorization_is_attested_but_never_replay_eligible(self) -> None:
        provider = self.provider()
        self.assertFalse(provider.replay_export_allowed)
        with self.assertRaisesRegex(
            src.StateProviderAuthorizationError, "never authorize replay"
        ):
            provider.require_replay_eligible()

    def test_production_remains_fail_closed_without_registered_verifier_digest(self) -> None:
        prerequisites = self.prerequisites(
            status=src.PRODUCTION_VERIFICATION_STATUS
        )
        self.assertIsNone(src.REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256)
        with self.assertRaisesRegex(
            src.StateProviderAuthorizationError, "no reviewed"
        ):
            src.verify_production_state_provider_prerequisites(prerequisites)

        forged = src.StateProviderAuthorizationV1(
            authorization_class=src.StateProviderAuthorizationClass.VERIFIED_EMPIRICAL,
            prerequisites_sha256=prerequisites.canonical_sha256,
            verifier_report_sha256=prerequisites.verifier_report_sha256,
        )
        with self.assertRaisesRegex(
            src.StateProviderAuthorizationError, "not verifier-attested"
        ):
            src.Run4ProductionStateProviderV1(
                prerequisites=prerequisites,
                authorization=forged,
            )

    def test_scaling_freshness_and_fit_bindings_change_provider_identity(self) -> None:
        first = self.provider().binding.canonical_sha256
        changed_prerequisites = self.prerequisites(max_age_ns=200_000_000)
        changed = src.Run4ProductionStateProviderV1(
            prerequisites=changed_prerequisites,
            authorization=src.authorize_test_only_state_provider(
                changed_prerequisites
            ),
        ).binding.canonical_sha256
        self.assertNotEqual(first, changed)

    def test_import_is_pure_and_cpu_only(self) -> None:
        script = r'''
import builtins
import json
import os
import socket
import subprocess
import sys

def blocked(*args, **kwargs):
    raise AssertionError("runtime side effect during import")

builtins.open = blocked
os.system = blocked
os.popen = blocked
subprocess.Popen = blocked
socket.socket = blocked
before = set(sys.modules)
import rl_agent.splitfusion_hybrid_sac_run4_v1.production_state_provider as module
after = set(sys.modules)
print(json.dumps({
    "registered": module.REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256,
    "torch": "torch" in after - before,
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(result.stdout)
        self.assertIsNone(observed["registered"])
        self.assertFalse(observed["torch"])


if __name__ == "__main__":
    unittest.main()

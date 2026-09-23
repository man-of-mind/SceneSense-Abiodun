"""Phase-1 CPU-only tests for the live Route-B frozen-policy runtime.

These tests read the pre-registered Run-3 artifacts and nothing else.  They
launch no CARLA, OAI, Docker or network process, and they assert that CUDA is
never initialized.  Every observation is an explicitly labelled synthetic
fixture; nothing here is measured evidence.

By default the expensive 131 MB full-checkpoint cross-verification runs once,
in :class:`CheckpointChainTest`.  Set
``SPLITFUSION_PILOT_SKIP_FULL_CHECKPOINT=1`` to skip it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    Q_E4_MAX,
    SPATIAL_CELLS,
    default_contract,
    keep_drop_counts,
    round_half_up_q_e4,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    StaleTelemetryError,
)

from . import analyzer, evidence
from . import pilot_contract as contract
from . import synthetic_fixtures as fixtures
from .checkpoint_loader import (
    CheckpointLoadError,
    load_pilot_actor_weights,
    project_root,
)
from .execution_identity import (
    MEASURED_ANCHOR,
    bind_execution_bundle,
    UNMEASURED_OFF_ANCHOR,
    ExecutionIdentityError,
    resolve_pilot_execution_identity,
)
from .frozen_actor import FrozenActorError, FrozenPilotActor
from .state_builder import (
    LivePolicyStateBuilderV1,
    SequentialObservationRefusedError,
    StateBuilderError,
)

_SKIP_FULL = os.environ.get("SPLITFUSION_PILOT_SKIP_FULL_CHECKPOINT") == "1"

_WEIGHTS = None
_ACTOR = None
_CONTRACT = None


def _weights():
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = load_pilot_actor_weights(cross_verify_full_checkpoint=False)
    return _WEIGHTS


def _actor() -> FrozenPilotActor:
    global _ACTOR
    if _ACTOR is None:
        _ACTOR = FrozenPilotActor(_weights())
    return _ACTOR


def _catalog():
    global _CONTRACT
    if _CONTRACT is None:
        _CONTRACT = default_contract()
    return _CONTRACT


def _genesis_features(**overrides) -> tuple:
    """A synthetic 31-D vector whose previous-outcome block is all zeros."""
    named = {name: 0.0 for name in POLICY_FEATURE_ORDER}
    named.update(
        {
            "scene_camera_si_scaled": 0.5,
            "scene_radar_p40": 0.4,
            "radio_achieved_snr_db_scaled": 0.6,
            "radio_mcs_index_scaled": 0.75,
        }
    )
    named.update(overrides)
    return tuple(named[name] for name in POLICY_FEATURE_ORDER)


# --------------------------------------------------------------------------- #


class PilotContractTest(unittest.TestCase):
    def test_training_support_covers_every_registered_feature_exactly_once(self):
        self.assertEqual(
            sorted(contract.TRAINING_SUPPORT), sorted(POLICY_FEATURE_ORDER)
        )
        self.assertEqual(len(contract.TRAINING_SUPPORT), POLICY_FEATURE_COUNT)

    def test_twenty_seven_features_are_declared_constant_zero(self):
        constants = [
            name
            for name, item in contract.TRAINING_SUPPORT.items()
            if item.exact_constant
        ]
        self.assertEqual(len(constants), 27)
        self.assertEqual(
            len(contract.POLICY_CONTROLLED_ZERO_FEATURES)
            + len(contract.MEASURED_LIVE_FEATURES),
            POLICY_FEATURE_COUNT,
        )

    def test_the_declared_varying_features_are_exactly_the_four_expected(self):
        varying = sorted(
            name
            for name, item in contract.TRAINING_SUPPORT.items()
            if not item.exact_constant
        )
        self.assertEqual(
            varying,
            [
                "radio_achieved_snr_db_scaled",
                "radio_mcs_index_scaled",
                "scene_camera_si_scaled",
                "scene_radar_p40",
            ],
        )

    def test_every_support_interval_names_registered_provenance(self):
        for name, item in contract.TRAINING_SUPPORT.items():
            with self.subTest(feature=name):
                self.assertIn(item.provenance, contract.SUPPORT_PROVENANCE)

    def test_contract_digest_is_reproducible(self):
        self.assertEqual(
            contract.PILOT_CONTRACT_SHA256,
            contract.canonical_sha256(contract.contract_descriptor()),
        )


class CheckpointChainTest(unittest.TestCase):
    def test_loader_verifies_the_whole_campaign_chain(self):
        weights = _weights()
        self.assertEqual(weights.seed, contract.PREREGISTERED_SEED)
        self.assertEqual(weights.update, contract.PREREGISTERED_UPDATE)
        self.assertEqual(
            weights.snapshot_file_sha256, contract.SNAPSHOT_FILE_SHA256
        )
        self.assertEqual(
            weights.runner_binding_sha256,
            contract.REGISTERED_RUNNER_BINDING_SHA256,
        )
        for link in (
            "campaign_complete.json:seed_reports[17]",
            "RUN3_TRAINING_COMPLETE.json:terminal_sha256",
            "report.json:report_sha256",
            "module_pin:snapshot_file_sha256",
            "model_snapshot:snapshot_sha256",
        ):
            self.assertIn(link, weights.verification_chain)

    def test_loader_refuses_a_missing_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CheckpointLoadError):
                load_pilot_actor_weights(
                    root=Path(directory), cross_verify_full_checkpoint=False
                )

    def test_loader_refuses_a_tampered_snapshot(self):
        base = project_root()
        source = base / contract.SNAPSHOT_RELATIVE_PATH
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory)
            target = fake_root / contract.SNAPSHOT_RELATIVE_PATH
            target.parent.mkdir(parents=True)
            payload = bytearray(source.read_bytes())
            payload[-1] ^= 0xFF
            target.write_bytes(bytes(payload))
            for relative in (
                contract.CAMPAIGN_COMPLETE_RELATIVE_PATH,
                contract.REPORT_RELATIVE_PATH,
                contract.TERMINAL_MARKER_RELATIVE_PATH,
            ):
                destination = fake_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((base / relative).read_bytes())
            with self.assertRaises(CheckpointLoadError):
                load_pilot_actor_weights(
                    root=fake_root, cross_verify_full_checkpoint=False
                )

    @unittest.skipIf(_SKIP_FULL, "full-checkpoint cross-verification disabled")
    def test_snapshot_actor_is_bitwise_equal_to_the_full_checkpoint(self):
        weights = load_pilot_actor_weights(cross_verify_full_checkpoint=True)
        self.assertTrue(weights.full_checkpoint_cross_verified)
        self.assertEqual(
            weights.full_checkpoint_file_sha256,
            contract.FULL_CHECKPOINT_FILE_SHA256,
        )
        self.assertIn(
            "full_checkpoint:bitwise_actor_equality", weights.verification_chain
        )
        self.assertFalse(torch.cuda.is_initialized())


class FrozenActorTest(unittest.TestCase):
    def test_repeated_decisions_are_bit_identical(self):
        actor = _actor()
        features = _genesis_features()
        first = actor.decide(features)
        for _ in range(8):
            other = actor.decide(features)
            self.assertEqual(first.mode_id, other.mode_id)
            self.assertEqual(first.q_e4, other.q_e4)
            self.assertEqual(first.q, other.q)
            self.assertEqual(first.mode_logits, other.mode_logits)

    def test_a_fresh_actor_instance_reproduces_the_same_decision(self):
        features = _genesis_features(scene_radar_p40=0.9)
        first = _actor().decide(features)
        second = FrozenPilotActor(_weights()).decide(features)
        self.assertEqual(
            (first.mode_id, first.q_e4), (second.mode_id, second.q_e4)
        )

    def test_feature_order_is_load_bearing(self):
        actor = _actor()
        baseline = actor.decide(_genesis_features())
        swapped = list(_genesis_features())
        index_si = POLICY_FEATURE_ORDER.index("scene_camera_si_scaled")
        index_snr = POLICY_FEATURE_ORDER.index("radio_achieved_snr_db_scaled")
        swapped[index_si], swapped[index_snr] = (
            swapped[index_snr],
            swapped[index_si],
        )
        permuted = actor.decide(tuple(swapped))
        self.assertNotEqual(
            (baseline.mode_id, baseline.q_e4), (permuted.mode_id, permuted.q_e4)
        )

    def test_wrong_width_is_refused(self):
        actor = _actor()
        for width in (0, POLICY_FEATURE_COUNT - 1, POLICY_FEATURE_COUNT + 1):
            with self.subTest(width=width):
                with self.assertRaises(FrozenActorError):
                    actor.decide(tuple([0.0] * width))

    def test_identifiers_cannot_reach_the_policy_tensor(self):
        actor = _actor()
        named = dict(zip(POLICY_FEATURE_ORDER, _genesis_features()))
        with self.assertRaises(FrozenActorError):
            actor.decide(named)
        with self.assertRaises(FrozenActorError):
            actor.decide("0" * POLICY_FEATURE_COUNT)
        with self.assertRaises(FrozenActorError):
            actor.decide(
                _genesis_features()[:-1] + ("6dc674b2-4fd5-5720-84b4-e59ffe1330a1",)
            )
        with self.assertRaises(FrozenActorError):
            actor.decide(_genesis_features()[:-1] + (5_000_000_000,))

    def test_non_finite_features_are_refused(self):
        actor = _actor()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                with self.assertRaises(FrozenActorError):
                    actor.decide(_genesis_features(scene_radar_p40=bad))

    def test_no_forbidden_identifier_substring_is_a_policy_feature(self):
        from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (  # noqa: E501
            FORBIDDEN_POLICY_FEATURE_SUBSTRINGS,
        )

        for name in POLICY_FEATURE_ORDER:
            for forbidden in FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
                self.assertNotIn(forbidden, name)

    def test_decision_mutates_no_ambient_rng(self):
        actor = _actor()
        random.seed(1234)
        torch.manual_seed(4321)
        np.random.seed(999)
        before = (
            random.getstate(),
            torch.get_rng_state().clone(),
            np.random.get_state(),
        )
        for _ in range(4):
            actor.decide(_genesis_features())
        after = (
            random.getstate(),
            torch.get_rng_state().clone(),
            np.random.get_state(),
        )
        self.assertEqual(before[0], after[0])
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertEqual(before[2][0], after[2][0])
        self.assertTrue((before[2][1] == after[2][1]).all())
        self.assertEqual(before[2][2:], after[2][2:])

    def test_no_cuda_initialization(self):
        _actor().decide(_genesis_features())
        self.assertFalse(torch.cuda.is_initialized())

    def test_actor_is_frozen_and_accumulates_no_gradient(self):
        actor = _actor()
        actor.decide(_genesis_features())
        actor.assert_frozen()
        self.assertFalse(actor._actor.training)
        for parameter in actor._actor.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_proposed_q_lies_in_the_registered_executable_interval(self):
        actor = _actor()
        for si in (0.0, 0.25, 0.5, 0.75, 1.0):
            for p40 in (0.0, 0.5, 1.0):
                proposal = actor.decide(
                    _genesis_features(
                        scene_camera_si_scaled=si, scene_radar_p40=p40
                    )
                )
                low, high = actor.support_bounds(proposal.mode_id)
                self.assertGreaterEqual(proposal.q_e4, low)
                self.assertLessEqual(proposal.q_e4, high)

    def test_proposal_records_hashes_and_inference_time(self):
        proposal = _actor().decide(_genesis_features())
        self.assertEqual(proposal.seed, contract.PREREGISTERED_SEED)
        self.assertEqual(proposal.update, contract.PREREGISTERED_UPDATE)
        self.assertEqual(
            proposal.snapshot_file_sha256, contract.SNAPSHOT_FILE_SHA256
        )
        self.assertEqual(
            proposal.actor_state_sha256, _weights().actor_state_sha256
        )
        self.assertGreater(proposal.inference_ns, 0)
        self.assertEqual(proposal.pilot_label, contract.PILOT_LABEL)


class ExecutionIdentityTest(unittest.TestCase):
    def test_off_anchor_proposal_carries_null_anchor_identity(self):
        proposal = _actor().decide(_genesis_features())
        resolved = resolve_pilot_execution_identity(proposal, _catalog())
        if resolved.is_registered_anchor:
            self.skipTest("the deterministic proposal happened to be an anchor")
        self.assertIsNone(resolved.action_id)
        self.assertIsNone(resolved.profile_id)
        self.assertEqual(resolved.measurement_status, UNMEASURED_OFF_ANCHOR)

    def test_every_registered_anchor_reproduces_its_catalog_identity(self):
        catalog = _catalog()
        for anchor in catalog.anchors:
            with self.subTest(profile=anchor.profile_id):
                action = catalog.resolve(anchor.mode.mode_id, anchor.q)
                self.assertEqual(action.q_e4, anchor.q_e4)
                self.assertEqual(action.action_id, anchor.action_id)
                self.assertEqual(action.profile_id, anchor.profile_id)
                self.assertEqual(action.keep_count, anchor.keep_count)
                self.assertEqual(action.drop_count, anchor.drop_count)

    def test_representative_off_anchor_values_keep_exact_counts(self):
        catalog = _catalog()
        anchor_q_e4 = {anchor.q_e4 for anchor in catalog.anchors}
        for mode_id in range(catalog.mode_count):
            for q_e4 in (1, 1234, 4999, 5001, 7777, 9799):
                if q_e4 in anchor_q_e4:
                    continue
                with self.subTest(mode=mode_id, q_e4=q_e4):
                    action = catalog.resolve(mode_id, q_e4 / 10000.0)
                    self.assertEqual(action.q_e4, q_e4)
                    keep, drop = keep_drop_counts(q_e4)
                    self.assertEqual((action.keep_count, action.drop_count), (keep, drop))
                    self.assertEqual(keep + drop, SPATIAL_CELLS)
                    self.assertIsNone(action.action_id)
                    self.assertIsNone(action.profile_id)

    def test_quantization_boundary_is_recomputed_not_trusted(self):
        proposal = _actor().decide(_genesis_features())
        forged = replace(proposal, q_e4=proposal.q_e4 + 1)
        with self.assertRaises(ExecutionIdentityError):
            resolve_pilot_execution_identity(forged, _catalog())

    def test_resolved_identity_serializes_and_is_self_consistent(self):
        proposal = _actor().decide(_genesis_features())
        resolved = resolve_pilot_execution_identity(proposal, _catalog())
        document = resolved.to_canonical_dict()
        self.assertEqual(document["q_e4"], proposal.q_e4)
        self.assertEqual(document["mode_id"], proposal.mode_id)
        self.assertEqual(
            document["keep_count"] + document["drop_count"], SPATIAL_CELLS
        )
        self.assertIsNone(document["execution_bundle_sha256"])
        self.assertIn("PHASE1", document["bundle_binding_status"])
        self.assertEqual(
            (document["action_id"] is not None),
            document["measurement_status"] == MEASURED_ANCHOR,
        )
        self.assertEqual(
            resolved.canonical_sha256(),
            contract.canonical_sha256(document),
        )

    def test_half_up_rounding_is_the_registered_rule(self):
        self.assertEqual(round_half_up_q_e4(0.12345), 1235)
        self.assertEqual(round_half_up_q_e4(1.5), Q_E4_MAX)

    def test_phase3_bundle_seam_binds_only_an_agreeing_profile(self):
        """The Phase-3 seam exists and already refuses a disagreeing profile."""

        @dataclasses.dataclass(frozen=True)
        class _StubDispatchProfile:
            mode_id: int
            q_e4: int
            keep_count: int
            drop_count: int
            action_id: object
            profile_id: object
            measurement_status: str
            execution_bundle_sha256: str

        proposal = _actor().decide(_genesis_features())
        resolved = resolve_pilot_execution_identity(proposal, _catalog())
        self.assertIsNone(resolved.execution_bundle_sha256)

        agreeing = _StubDispatchProfile(
            mode_id=resolved.mode_id,
            q_e4=resolved.q_e4,
            keep_count=resolved.keep_count,
            drop_count=resolved.drop_count,
            action_id=resolved.action_id,
            profile_id=resolved.profile_id,
            measurement_status=resolved.measurement_status,
            execution_bundle_sha256="a" * 64,
        )
        bound = bind_execution_bundle(resolved, agreeing)
        self.assertEqual(bound.execution_bundle_sha256, "a" * 64)
        self.assertEqual(
            bound.bundle_binding_status, "PHASE3_DYNAMIC_EXECUTION_CONTRACT_BOUND"
        )
        self.assertEqual(bound.q_e4, resolved.q_e4)
        self.assertEqual(bound.keep_count, resolved.keep_count)

        disagreeing = dataclasses.replace(agreeing, q_e4=resolved.q_e4 + 1)
        with self.assertRaises(ExecutionIdentityError):
            bind_execution_bundle(resolved, disagreeing)

        without_digest = dataclasses.replace(agreeing, execution_bundle_sha256="")
        with self.assertRaises(ExecutionIdentityError):
            bind_execution_bundle(resolved, without_digest)


class StateBuilderTest(unittest.TestCase):
    OBSERVED_NS = 5_000_000_000

    def _build(self, **overrides):
        session = fixtures.synthetic_session_uuid("state")
        lineage = fixtures.synthetic_session_uuid("state-lineage")
        observed = overrides.pop("observed_ns", self.OBSERVED_NS)
        frame = overrides.pop("carla_frame_id", 42)
        scene_age = overrides.pop("scene_age_ns", 20_000_000)
        radio_age = overrides.pop("radio_age_ns", 10_000_000)
        builder = overrides.pop("builder", None) or LivePolicyStateBuilderV1()
        scene = overrides.pop("scene", None) or fixtures.synthetic_scene(
            camera_si=overrides.pop("camera_si", 110.0),
            radar_p40=overrides.pop("radar_p40", 0.3),
            measured_ns=observed - scene_age,
            carla_frame_id=frame,
        )
        radio = overrides.pop("radio", None) or fixtures.synthetic_radio(
            achieved_snr_db=overrides.pop("snr_db", 18.0),
            mcs_index=overrides.pop("mcs_index", 20),
            bsr_bytes=overrides.pop("bsr_bytes", 0),
            measured_ns=observed - radio_age,
            session_uuid=session,
        )
        start = fixtures.synthetic_episode_start(
            session_uuid=session,
            lineage_uuid=lineage,
            tensor_seq=0,
            carla_frame_id=frame,
            observed_ns=observed,
        )
        return builder.build(
            scene=scene,
            radio=radio,
            episode_start=start,
            session_uuid=session,
            observed_ns=observed,
            tensor_seq=0,
            carla_frame_id=frame,
            **overrides,
        )

    def test_uses_the_registered_run3_preprocessing_by_default(self):
        builder = LivePolicyStateBuilderV1()
        self.assertTrue(builder.uses_registered_run3_preprocessing)
        self.assertEqual(
            builder.normalization_spec_sha256,
            "9cb462f72c5d561b5f766f9ececee2fd0890af2d90013533d9786f8ed962a842",
        )
        self.assertEqual(
            builder.freshness_policy_sha256,
            "5ea46607354a83fcf32cd8f09e68df71b62d9c2ba0abdd29d2a817048a7dec10",
        )

    def test_builds_a_genesis_matched_31_feature_vector(self):
        built = self._build()
        self.assertEqual(len(built.values), POLICY_FEATURE_COUNT)
        self.assertTrue(built.genesis_matched)
        named = built.as_mapping()
        for name in contract.POLICY_CONTROLLED_ZERO_FEATURES:
            self.assertEqual(named[name], 0.0)

    def test_ages_are_derived_not_asserted(self):
        built = self._build(scene_age_ns=30_000_000, radio_age_ns=12_000_000)
        self.assertEqual(built.measurement_ages_ns["scene"], 30_000_000)
        self.assertEqual(built.measurement_ages_ns["snr"], 12_000_000)
        named = built.as_mapping()
        self.assertAlmostEqual(named["freshness_scene_normalized"], 0.30)
        self.assertAlmostEqual(named["freshness_snr_normalized"], 0.12)

    def test_a_future_measurement_is_refused(self):
        with self.assertRaises(StateBuilderError):
            self._build(scene_age_ns=-1_000_000)

    def test_a_stale_measurement_fails_closed_and_substitutes_nothing(self):
        with self.assertRaises(StaleTelemetryError):
            self._build(scene_age_ns=150_000_000)

    def test_a_chained_previous_outcome_is_refused(self):
        sentinel = object()
        with self.assertRaises(SequentialObservationRefusedError):
            self._build(previous=sentinel)

    def test_the_builder_never_manufactures_radio_evidence(self):
        with self.assertRaises(StateBuilderError):
            self._build(radio=object())

    def test_support_audit_flags_live_ages_as_out_of_support(self):
        built = self._build(scene_age_ns=30_000_000, radio_age_ns=12_000_000)
        flagged = set(built.support_audit.out_of_support_features)
        self.assertIn("freshness_scene_normalized", flagged)
        self.assertIn("freshness_snr_normalized", flagged)
        self.assertFalse(built.support_audit.fully_in_support)

    def test_support_audit_is_clean_for_a_genesis_matched_observation(self):
        built = self._build(scene_age_ns=0, radio_age_ns=0, bsr_bytes=0)
        self.assertTrue(
            built.support_audit.fully_in_support,
            built.support_audit.out_of_support_features,
        )

    def test_a_non_empty_uplink_buffer_is_recorded_not_zeroed(self):
        built = self._build(scene_age_ns=0, radio_age_ns=0, bsr_bytes=4096)
        named = built.as_mapping()
        self.assertEqual(built.state.radio.bsr_bytes, 4096)
        self.assertNotEqual(named["radio_bsr_log1p_scaled"], 0.0)
        self.assertIn(
            "radio_bsr_log1p_scaled", built.support_audit.out_of_support_features
        )

    def test_the_registered_bsr_scaling_saturates_on_any_live_buffer(self):
        """The registered scale is 1.0, so log1p saturates above ~1.72 bytes.

        This is a property of the *registered* spec, not of this pilot: its own
        provenance records ``INERT_FOR_EXACT_GENESIS_ZERO_ONLY``.  Pinning it
        here makes the consequence visible rather than surprising.
        """
        for byte_count in (2, 1024, 4096, 1_000_000):
            built = self._build(
                scene_age_ns=0, radio_age_ns=0, bsr_bytes=byte_count
            )
            self.assertEqual(built.as_mapping()["radio_bsr_log1p_scaled"], 1.0)

    def test_the_built_vector_drives_the_frozen_actor(self):
        built = self._build(scene_age_ns=0, radio_age_ns=0)
        proposal = _actor().decide(built.values)
        resolved = resolve_pilot_execution_identity(proposal, _catalog())
        self.assertEqual(resolved.q_e4, proposal.q_e4)

    def test_state_and_feature_digests_are_reproducible(self):
        first = self._build()
        second = self._build()
        self.assertEqual(first.state_sha256, second.state_sha256)
        self.assertEqual(first.features_sha256, second.features_sha256)


class EvidenceTest(unittest.TestCase):
    def test_writer_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "session"
            writer = evidence.PilotEvidenceWriter(target)
            writer.close()
            with self.assertRaises(evidence.EvidenceError):
                evidence.PilotEvidenceWriter(target)

    def test_blank_record_is_complete_and_validates(self):
        record = evidence.blank_frame_record()
        evidence.validate_frame_record(record)
        for group in evidence.FRAME_FIELD_GROUPS.values():
            for name in group:
                self.assertIn(name, record)

    def test_absent_class_metrics_are_null_not_zero(self):
        record = evidence.blank_frame_record()
        for name in (
            "loc_person_recall",
            "loc_person_xy_error_m",
            "loc_person_footprint_iou",
            "seg_person_iou",
        ):
            self.assertIsNone(record[name])
        for name in (
            "loc_person_status",
            "loc_vehicle_status",
            "seg_person_status",
            "seg_vehicle_status",
        ):
            self.assertIsInstance(record[name], str)
            self.assertTrue(record[name])

    def test_field_drift_is_refused(self):
        record = evidence.blank_frame_record()
        record["an_unregistered_field"] = 1
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_frame_record(record)
        del record["an_unregistered_field"]
        del record["q_perc"]
        with self.assertRaises(evidence.EvidenceError):
            evidence.validate_frame_record(record)

    def test_session_header_is_written_once(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = evidence.PilotEvidenceWriter(Path(directory) / "s")
            writer.write_session({"run_id": "phase1-selftest"})
            with self.assertRaises(evidence.EvidenceError):
                writer.write_session({"run_id": "phase1-selftest"})
            writer.close()


class AnalyzerTest(unittest.TestCase):
    def _write_session(self, directory: Path) -> None:
        actor = _actor()
        builder = LivePolicyStateBuilderV1()
        session = fixtures.synthetic_session_uuid("analyzer")
        lineage = fixtures.synthetic_session_uuid("analyzer-lineage")
        writer = evidence.PilotEvidenceWriter(directory)
        writer.write_session(
            {
                "run_id": "phase1-analyzer-selftest",
                "actor_state_sha256": actor.actor_state_sha256,
            }
        )
        observed = 7_000_000_000
        for index in range(3):
            frame = 100 + index
            built = builder.build(
                scene=fixtures.synthetic_scene(
                    camera_si=100.0 + index,
                    radar_p40=0.2 + 0.1 * index,
                    measured_ns=observed,
                    carla_frame_id=frame,
                ),
                radio=fixtures.synthetic_radio(
                    achieved_snr_db=12.0 + index,
                    mcs_index=15 + index,
                    bsr_bytes=0,
                    measured_ns=observed,
                    session_uuid=session,
                    index=index,
                ),
                episode_start=fixtures.synthetic_episode_start(
                    session_uuid=session,
                    lineage_uuid=lineage,
                    tensor_seq=index,
                    carla_frame_id=frame,
                    observed_ns=observed,
                ),
                session_uuid=session,
                observed_ns=observed,
                tensor_seq=index,
                carla_frame_id=frame,
            )
            proposal = actor.decide(built.values)
            resolved = resolve_pilot_execution_identity(proposal, _catalog())
            record = evidence.blank_frame_record()
            record.update(
                {
                    "session_uuid": session,
                    "controller_lineage_uuid": lineage,
                    "carla_frame_id": frame,
                    "tensor_seq": index,
                    "opened_decision": True,
                    "reused_held_action": False,
                    "reward_requested": True,
                    "policy_features": list(built.values),
                    "support_audit": built.support_audit.to_canonical_dict(),
                    "state_sha256": built.state_sha256,
                    "features_sha256": built.features_sha256,
                    "proposed_mode_id": proposal.mode_id,
                    "proposed_q": proposal.q,
                    "executed_mode_id": resolved.mode_id,
                    "executed_q_e4": resolved.q_e4,
                    "executed_action_id": resolved.action_id,
                    "executed_profile_id": resolved.profile_id,
                    "executed_measurement_status": resolved.measurement_status,
                    "executed_keep_count": resolved.keep_count,
                    "executed_drop_count": resolved.drop_count,
                    "execution_identity_sha256": resolved.canonical_sha256(),
                    "policy_inference_ns": proposal.inference_ns,
                    "actor_state_sha256": proposal.actor_state_sha256,
                }
            )
            writer.write_frame(record)
        writer.close({"status": "PHASE1_SELFTEST_COMPLETE"})

    def test_analyzer_decides_phase1_gates_and_abstains_on_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "session"
            self._write_session(target)
            result = analyzer.analyze_session(
                target, expected_actor_state_sha256=_actor().actor_state_sha256
            )
            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.decision_count, 3)
            self.assertEqual(result.failed, ())
            self.assertEqual(len(result.passed), 5)
            self.assertEqual(len(result.not_evaluable), 5)
            for gate_id, _ in analyzer.GATE_INVENTORY:
                self.assertIn(
                    gate_id, result.passed + result.not_evaluable + result.failed
                )

    def test_analyzer_fails_a_chained_previous_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "session"
            self._write_session(target)
            path = target / evidence.PilotEvidenceWriter.FRAME_FILENAME
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            index = POLICY_FEATURE_ORDER.index("prev_present_mask")
            rows[1]["policy_features"][index] = 1.0
            path.write_text(
                "\n".join(
                    contract.canonical_json_bytes(row).decode() for row in rows
                )
                + "\n"
            )
            result = analyzer.analyze_session(target)
            self.assertIn("G04_GENESIS_MATCHED_STATE", result.failed)

    def test_analyzer_never_writes_into_the_session(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "session"
            self._write_session(target)
            before = sorted(p.name for p in target.iterdir())
            analyzer.analyze_session(target)
            self.assertEqual(before, sorted(p.name for p in target.iterdir()))


class ProcessIsolationTest(unittest.TestCase):
    def test_importing_the_package_initializes_nothing(self):
        import subprocess
        import sys

        code = (
            "import sys;"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v1 as p;"
            "assert p.PACKAGE_LABEL.startswith('GENESIS_MATCHED');"
            "assert 'torch' not in sys.modules, 'importing the package pulled in torch';"
            "print('OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_cuda_is_never_initialized_by_the_phase1_path(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

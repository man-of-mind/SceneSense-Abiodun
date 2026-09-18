"""Tests for the Phase-4a (4a.1) causal-state, reward and transition contract.

Every value is deterministic and injected: no clock is read, nothing sleeps, and
no CARLA, OAI, Docker, CUDA or network resource is touched.  ACK documents are
built with the **real** protocol builders and verified through the **real**
validator, so the binding tests exercise the production wire contract rather
than a local imitation.

Structure
---------
``BaseContractTest``      shared builders.
``ContractBehaviourTest`` the positive contract and its numeric properties.
``AdversarialRejectionTest``
                          the fourteen required rejection proofs.

Canonical hashes are recomputed with a locally written canonicalizer rather than
by calling the module's own helper.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from . import action_contract as ac
from . import reward_ticket_controller as rtc
from . import scene_descriptors as sd
from . import state_reward_transition_contract as src
from . import transaction_identity as ti

SESSION = "3f263fce-cc44-476e-93b5-19d09d439471"
OTHER_SESSION = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"

MS = 1_000_000
T0 = 1_000_000_000
B = rtc.B_REWARD_DEADLINE_NS
WALL0 = 1_700_000_000_000_000_000

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _independent_canonical_bytes(payload: Any) -> bytes:
    """A locally written canonicalizer, independent of the module under test."""
    return json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _independent_sha256(payload: Any) -> str:
    return hashlib.sha256(_independent_canonical_bytes(payload)).hexdigest()


class BaseContractTest(unittest.TestCase):
    """Shared deterministic builders for the Phase-4a contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()
        cls.protocol = src._load_quality_protocol()

    # -- actions ----------------------------------------------------------- #

    def _action(
        self, mode_id: int = 3, q_e4: int = 9800
    ) -> ti.ExecutedActionIdentity:
        executable = self.contract.resolve(mode_id, q_e4 / ac.Q_E4_SCALE)
        return ti.ExecutedActionIdentity.from_executable_action(
            executable, self.contract
        )

    def _off_anchor_action(self) -> ti.ExecutedActionIdentity:
        action = self._action(mode_id=3, q_e4=4321)
        assert action.action_id is None
        return action

    # -- specs ------------------------------------------------------------- #

    def _reward_spec(self, **overrides: Any) -> src.RewardSpecV1:
        kwargs: Dict[str, Any] = dict(
            spec_id="phase4a1_unit_test_reward",
            spec_version=1,
            w_loc_person=2.0,
            w_loc_vehicle=1.0,
            tau_person_m=1.5,
            tau_vehicle_m=3.0,
            localization_combiner=(
                src.LocalizationCombiner.WEIGHTED_GEOMETRIC_MEAN
            ),
            w_seg_person=2.0,
            w_seg_vehicle=1.0,
            seg_reference_person_iou=0.60,
            seg_reference_vehicle_iou=0.80,
            segmentation_modulation_beta=0.30,
            w_quality=1.0,
            w_latency=0.25,
            lambda_mode=0.05,
            lambda_q=0.10,
            r_registered_failure=-1.0,
            gamma_per_tensor=0.99,
            provenance={"origin": "unit test; every value is a hypothesis"},
        )
        kwargs.update(overrides)
        return src.RewardSpecV1(**kwargs)

    def _norm(self, **overrides: Any) -> src.StateNormalizationSpecV1:
        kwargs: Dict[str, Any] = dict(
            spec_id="phase4a1_unit_test_norm",
            spec_version=1,
            train_split_id="unit-test-synthetic-split",
            fit_population_count=1234,
            fit_config_sha256=HEX_A,
            camera_si_clip_min=0.0,
            camera_si_clip_max=100.0,
            achieved_snr_db_clip_min=-5.0,
            achieved_snr_db_clip_max=35.0,
            bsr_log1p_scale=math.log1p(1_000_000.0),
            snr_metric=src.SnrMetric.UL_PUSCH_POST_EQUALISER_SINR_DB,
            mcs_table_id="oai_ul_table_1",
            mcs_table_max_index=27,
            bsr_scope=src.BsrScope.ALL_GROUPS_LATEST,
            provenance={"origin": "unit test; not a fitted production spec"},
        )
        kwargs.update(overrides)
        return src.StateNormalizationSpecV1(**kwargs)

    def _freshness(self, **overrides: Any) -> src.StateFreshnessPolicyV1:
        kwargs: Dict[str, Any] = dict(
            policy_id="phase4a1_unit_test_freshness",
            max_scene_age_ns=150 * MS,
            max_snr_age_ns=200 * MS,
            max_bsr_age_ns=200 * MS,
            max_mcs_age_ns=200 * MS,
            provenance={"origin": "unit test bounds; not a measured budget"},
        )
        kwargs.update(overrides)
        return src.StateFreshnessPolicyV1(**kwargs)

    # -- observations and state -------------------------------------------- #

    def _scene(self, **overrides: Any) -> src.SceneObservationV1:
        kwargs: Dict[str, Any] = dict(
            sample=sd.SceneDescriptorSample(camera_si=42.5, radar_p40=0.375),
            measured_ns=T0 - 20 * MS,
            carla_frame_id=500,
            source_id="route_b_cell_rgb_radar_v1",
            source_sha256=HEX_B,
        )
        kwargs.update(overrides)
        return src.SceneObservationV1(**kwargs)

    def _radio(self, **overrides: Any) -> src.RadioObservationV1:
        kwargs: Dict[str, Any] = dict(
            achieved_snr_db=18.25,
            snr_metric=src.SnrMetric.UL_PUSCH_POST_EQUALISER_SINR_DB,
            snr_direction=src.LinkDirection.UPLINK,
            snr_measured_ns=T0 - 30 * MS,
            mcs_index=13,
            mcs_table_id="oai_ul_table_1",
            mcs_direction=src.LinkDirection.UPLINK,
            mcs_measured_ns=T0 - 35 * MS,
            bsr_bytes=4096,
            bsr_scope=src.BsrScope.ALL_GROUPS_LATEST,
            bsr_logical_channel_group=1,
            bsr_measured_ns=T0 - 25 * MS,
            source_id="oai_ue_mac_stats_v1",
            source_sha256=HEX_C,
        )
        kwargs.update(overrides)
        return src.RadioObservationV1(**kwargs)

    def _state(self, **overrides: Any) -> src.CausalStateV1:
        frame = overrides.pop("carla_frame_id", 500)
        kwargs: Dict[str, Any] = dict(
            scene=self._scene(carla_frame_id=frame),
            radio=self._radio(),
            session_uuid=SESSION,
            observed_ns=T0,
            tensor_seq=10,
            carla_frame_id=frame,
            previous=None,
        )
        kwargs.update(overrides)
        return src.CausalStateV1(**kwargs)

    # -- tickets ----------------------------------------------------------- #

    def _ticket(
        self,
        *,
        terminal: rtc.TerminalClass = rtc.TerminalClass.REWARD_FINAL_EXACT,
        decision_seq: int = 1,
        first_tensor_seq: int = 10,
        first_frame_id: int = 500,
        extra_reuses: int = 0,
        action: Optional[ti.ExecutedActionIdentity] = None,
        opened_ns: int = T0,
        resolution_offset_ns: int = 50 * MS,
        session_uuid: str = SESSION,
    ) -> rtc.CompletedTicket:
        """Drive the real Phase-3b controller to produce a genuine ticket."""
        action = action if action is not None else self._action()
        controller = rtc.RewardTicketController(session_uuid)
        controller.open_decision(
            decision_seq=decision_seq,
            tensor_seq=first_tensor_seq,
            carla_frame_id=first_frame_id,
            action=action,
            now_ns=opened_ns,
        )
        terminal_offset = (
            B + 1
            if terminal is rtc.TerminalClass.FEEDBACK_TIMEOUT
            else resolution_offset_ns
        )
        tensor_seq, frame_id = first_tensor_seq, first_frame_id
        for index in range(1 + extra_reuses):
            tensor_seq += 1
            frame_id += 1
            controller.reuse_held_action(
                tensor_seq=tensor_seq,
                carla_frame_id=frame_id,
                now_ns=opened_ns + min((index + 1) * 10 * MS, terminal_offset),
            )
        if terminal is rtc.TerminalClass.FEEDBACK_TIMEOUT:
            completed = controller.observe(opened_ns + B + 1).completed_ticket
        elif terminal is rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
            completed = controller.record_infrastructure_fault(
                decision_seq=decision_seq,
                now_ns=opened_ns + resolution_offset_ns,
                detail="unit-test evaluator fault",
            ).completed_ticket
        else:
            status = (
                rtc.FeedbackTerminalStatus.REWARD_FINAL
                if terminal is rtc.TerminalClass.REWARD_FINAL_EXACT
                else rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE
            )
            completed = controller.submit_feedback(
                rtc.RewardFeedbackMessage(
                    identity=ti.RewardFeedbackIdentity(
                        session_uuid=session_uuid,
                        decision_seq=decision_seq,
                        reward_tensor_seq=first_tensor_seq,
                        carla_frame_id=first_frame_id,
                        action=action,
                    ),
                    terminal_status=status,
                ),
                opened_ns + resolution_offset_ns,
            ).completed_ticket
        assert completed is not None
        assert completed.terminal_class is terminal
        return completed

    # -- ACK documents ----------------------------------------------------- #

    def _ack_quality(self, **overrides: Any) -> Dict[str, Any]:
        """The nested quality mapping the real ACK builder consumes."""
        vehicle: Dict[str, Any] = dict(
            recall=0.75,
            source_time_world_xy_error_m=0.95,
            footprint_iou=0.5,
            tp=3,
            fn=1,
        )
        person: Dict[str, Any] = dict(
            recall=0.5,
            source_time_world_xy_error_m=0.50,
            footprint_iou=0.4,
            tp=1,
            fn=1,
        )
        segmentation: Dict[str, Any] = dict(
            miou_vehicle_iou=0.80,
            miou_person_iou=0.60,
            miou_3class_macro=0.70,
            gt_vehicle_pixels=1200,
            gt_person_pixels=300,
        )
        vehicle.update(overrides.pop("vehicle", {}))
        person.update(overrides.pop("person", {}))
        segmentation.update(overrides.pop("segmentation", {}))
        assert not overrides, overrides
        return {
            "segmentation": segmentation,
            "localization": {"vehicle": vehicle, "person": person},
        }

    def _identity_fields(
        self,
        action: ti.ExecutedActionIdentity,
        *,
        frame_id: int = 500,
        run_id: str = "run-a",
        cell_id: str = "cell-a",
        stream_id: str = "stream-a",
        capture_timestamp_ns: int = WALL0,
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "cell_id": cell_id,
            "stream_id": stream_id,
            "frame_id": frame_id,
            "action_id": action.action_id,
            "profile_id": action.profile_id,
            "capture_timestamp_ns": capture_timestamp_ns,
        }

    def _ack_document(
        self,
        action: Optional[ti.ExecutedActionIdentity] = None,
        *,
        quality: Optional[Mapping[str, Any]] = None,
        evaluator_mode: str = "exact_carla_gt_v1",
        **identity_overrides: Any,
    ) -> Dict[str, Any]:
        """Build a genuine ACK document with the real protocol builders."""
        action = action if action is not None else self._action()
        identity = self._identity_fields(action, **identity_overrides)
        timing = {
            name: WALL0 + 1000 * (index + 1)
            for index, name in enumerate(self.protocol.TIMING_FIELDS)
        }
        payload = dict(quality) if quality is not None else self._ack_quality()
        detail = self.protocol.build_detail(
            identity_fields=identity,
            frozen_carla_frame_id=int(identity["frame_id"]),
            timing=timing,
            quality=payload,
            evaluator_mode=evaluator_mode,
        )
        return self.protocol.build_ack(
            identity_fields=identity,
            frozen_carla_frame_id=int(identity["frame_id"]),
            timing=timing,
            quality=payload,
            evaluator_mode=evaluator_mode,
            detail_sha256=self.protocol.detail_digest(detail),
        )

    def _obligation(
        self,
        ticket: rtc.CompletedTicket,
        **overrides: Any,
    ) -> src.QualityAckObligationV1:
        kwargs: Dict[str, Any] = dict(
            run_id="run-a",
            cell_id="cell-a",
            stream_id="stream-a",
            capture_timestamp_ns=WALL0,
        )
        kwargs.update(overrides)
        return src.QualityAckObligationV1.for_reward_tensor(ticket, **kwargs)

    def _eligibility(self, **overrides: Any) -> src.EvaluationEligibilityV1:
        kwargs: Dict[str, Any] = dict(
            ue_id="ue-1",
            eligibility_contract_id="route_b_range50_fov90_avo_v1",
            eligibility_contract_sha256=HEX_A,
            max_range_m=50.0,
            fov_deg=90.0,
            visibility_rule=src.VisibilityRule.AVO_ACTOR_VISIBLE_OBJECT,
            segmentation_eligibility_masked=True,
        )
        kwargs.update(overrides)
        return src.EvaluationEligibilityV1(**kwargs)

    def _binding(
        self,
        ticket: rtc.CompletedTicket,
        *,
        document: Optional[Mapping[str, Any]] = None,
        obligation: Optional[src.QualityAckObligationV1] = None,
        **kwargs: Any,
    ) -> src.QualityAckBindingV1:
        obligation = obligation if obligation is not None else self._obligation(
            ticket
        )
        document = (
            document
            if document is not None
            else self._ack_document(
                ticket.action, frame_id=ticket.reward_carla_frame_id
            )
        )
        return src.QualityAckBindingV1.from_ack_document(
            document, obligation=obligation, completed_ticket=ticket, **kwargs
        )

    def _evidence(
        self,
        ticket: rtc.CompletedTicket,
        *,
        obligation: Optional[src.QualityAckObligationV1] = None,
        binding: Optional[src.QualityAckBindingV1] = None,
        eligibility: Optional[src.EvaluationEligibilityV1] = None,
    ) -> src.QualityEvidenceV1:
        obligation = obligation if obligation is not None else self._obligation(
            ticket
        )
        binding = (
            binding
            if binding is not None
            else self._binding(ticket, obligation=obligation)
        )
        return src.QualityEvidenceV1.for_verified_ack(
            binding,
            obligation=obligation,
            eligibility=eligibility or self._eligibility(),
            gt_source_detail="carla_0_10_town10hd_opt_actor_origin_gt",
        )

    def _components(
        self,
        ticket: Optional[rtc.CompletedTicket] = None,
        *,
        evidence: Optional[src.QualityEvidenceV1] = None,
        pred_vehicle_pixels: int = 1100,
        pred_person_pixels: int = 280,
        vehicle_eligible_gt_instances: int = 4,
        person_eligible_gt_instances: int = 2,
    ) -> src.QualityComponentsV1:
        if evidence is None:
            assert ticket is not None
            evidence = self._evidence(ticket)
        return src.QualityComponentsV1.from_ack_binding(
            evidence,
            pred_vehicle_pixels=pred_vehicle_pixels,
            pred_person_pixels=pred_person_pixels,
            vehicle_eligible_gt_instances=vehicle_eligible_gt_instances,
            person_eligible_gt_instances=person_eligible_gt_instances,
        )

    def _trace(
        self, action: ti.ExecutedActionIdentity, **overrides: Any
    ) -> src.PolicyDecisionTraceV1:
        kwargs: Dict[str, Any] = dict(
            sampled_mode_id=action.mode_id,
            sampled_q=float(action.q_e4) / ac.Q_E4_SCALE,
            executed_action=action,
            log_prob_discrete=-1.25,
            log_prob_continuous=-0.75,
            actor_version_sha256=HEX_B,
        )
        kwargs.update(overrides)
        return src.PolicyDecisionTraceV1(**kwargs)

    def _adjudication(
        self,
        ticket: rtc.CompletedTicket,
        verdict: src.Adjudication,
        **overrides: Any,
    ) -> src.AdjudicationRecordV1:
        kwargs: Dict[str, Any] = dict(
            verdict=verdict,
            adjudicator_id="unit_test_reconciler",
            evidence_sha256=HEX_C,
            adjudicated_ns=ticket.closed_ns + 10 * B,
            detail=f"unit-test verdict {verdict.value}",
        )
        kwargs.update(overrides)
        return src.AdjudicationRecordV1.for_ticket(ticket, **kwargs)

    # -- whole transitions ------------------------------------------------- #

    def _next_state(
        self,
        ticket: rtc.CompletedTicket,
        outcome: src.DecisionOutcomeV1,
        spec: src.RewardSpecV1,
        **overrides: Any,
    ) -> src.CausalStateV1:
        previous = src.PreviousOutcomeV1.from_completed(ticket, outcome, spec)
        observed = overrides.pop("observed_ns", ticket.closed_ns + 50 * MS)
        frame = overrides.pop("carla_frame_id", 600)
        kwargs: Dict[str, Any] = dict(
            scene=self._scene(
                carla_frame_id=frame, measured_ns=observed - 10 * MS
            ),
            radio=self._radio(
                snr_measured_ns=observed - 10 * MS,
                bsr_measured_ns=observed - 10 * MS,
                mcs_measured_ns=observed - 10 * MS,
            ),
            session_uuid=ticket.session_uuid,
            observed_ns=observed,
            tensor_seq=ticket.tensor_seqs[-1] + 1,
            carla_frame_id=frame,
            previous=previous,
        )
        kwargs.update(overrides)
        return src.CausalStateV1(**kwargs)

    def _full_transition(
        self,
        *,
        terminal: rtc.TerminalClass = rtc.TerminalClass.REWARD_FINAL_EXACT,
        spec: Optional[src.RewardSpecV1] = None,
        extra_reuses: int = 0,
        previous_action: Optional[ti.ExecutedActionIdentity] = None,
        state_previous: Optional[src.PreviousOutcomeV1] = None,
    ) -> src.ReplayTransitionV1:
        spec = spec or self._reward_spec()
        norm, fresh = self._norm(), self._freshness()
        ticket = self._ticket(terminal=terminal, extra_reuses=extra_reuses)
        components = (
            self._components(ticket)
            if terminal is rtc.TerminalClass.REWARD_FINAL_EXACT
            else None
        )
        outcome = src.evaluate_completed_decision(
            ticket,
            spec,
            quality_components=components,
            previous_action=previous_action,
        )
        return src.build_replay_transition(
            state=self._state(previous=state_previous),
            next_state=self._next_state(ticket, outcome, spec),
            completed_ticket=ticket,
            outcome=outcome,
            policy_trace=self._trace(ticket.action),
            reward_spec=spec,
            normalization=norm,
            freshness=fresh,
        )


class ContractBehaviourTest(BaseContractTest):
    """The positive contract, its numeric properties and its bindings."""

    # ------------------------------------------------------------------ 1 -- #

    def test_ack_binding_is_verified_from_the_real_document(self) -> None:
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        document = self._ack_document(
            ticket.action, frame_id=ticket.reward_carla_frame_id
        )
        binding = src.QualityAckBindingV1.from_ack_document(
            document, obligation=obligation, completed_ticket=ticket
        )
        self.assertTrue(binding.is_attested)

        # the raw hash is recomputed from the document by the protocol itself
        self.assertEqual(
            binding.raw_quality_ack_sha256, self.protocol.digest(document)
        )
        self.assertEqual(binding.detailed_evidence_sha256, document["dh"])
        # all seven identity fields are retained verbatim
        self.assertEqual(
            tuple(sorted(binding.identity_fields)),
            tuple(sorted(src.QUALITY_ACK_IDENTITY_FIELDS)),
        )
        self.assertEqual(
            dict(binding.identity_fields),
            dict(self.protocol.identity_dict(document)),
        )
        # and all fifteen score fields
        self.assertEqual(
            tuple(sorted(binding.quality_fields)),
            tuple(sorted(src.QUALITY_ACK_QUALITY_FIELDS)),
        )
        self.assertEqual(binding.action_id, ticket.action.action_id)
        self.assertEqual(binding.profile_id, ticket.action.profile_id)
        self.assertEqual(binding.frame_id, ticket.reward_carla_frame_id)
        self.assertEqual(binding.obligation_sha256, obligation.canonical_sha256())
        self.assertEqual(
            binding.completed_ticket_sha256, ticket.canonical_sha256()
        )
        # an agreeing caller hash is accepted as a cross-check
        src.QualityAckBindingV1.from_ack_document(
            document,
            obligation=obligation,
            completed_ticket=ticket,
            expected_raw_ack_sha256=self.protocol.digest(document),
        )
        # the obligation binds the complete executed action
        self.assertEqual(
            obligation.executed_action_sha256, ticket.action.canonical_sha256()
        )
        self.assertTrue(obligation.is_anchor_expressible)
        for field in (
            "run_id", "cell_id", "stream_id", "frame_id",
            "capture_timestamp_ns", "session_uuid", "decision_seq",
            "reward_tensor_seq",
        ):
            with self.subTest(field=field):
                self.assertIn(field, obligation.to_canonical_dict())

    # ------------------------------------------------------------------ 2 -- #

    def test_schema_binds_all_dependencies_including_the_protocol(self) -> None:
        descriptor = src.SCHEMA_DESCRIPTOR
        self.assertIsInstance(descriptor, MappingProxyType)
        self.assertEqual(src.SCHEMA_VERSION, 2)
        self.assertEqual(src.SCHEMA_SHA256, _independent_sha256(descriptor))
        self.assertIn("phase 4a.1", descriptor["revision_note"])

        deps = descriptor["dependencies"]
        self.assertEqual(deps["action_catalog"]["sha256"], ac.CATALOG_SHA256)
        self.assertEqual(
            deps["action_catalog"]["sha256"],
            "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3",
        )
        self.assertEqual(
            deps["transaction_identity"]["schema_sha256"], ti.SCHEMA_SHA256
        )
        self.assertEqual(
            deps["scene_descriptors"]["schema_sha256"], sd.SCHEMA_SHA256
        )
        self.assertEqual(
            deps["reward_ticket_controller"]["schema_sha256"],
            rtc.CONTROLLER_SCHEMA_SHA256,
        )
        self.assertEqual(deps["reward_ticket_controller"]["schema_version"], 2)
        # the complete quality-protocol contract is bound by hash
        self.assertEqual(
            deps["quality_protocol"]["contract_sha256"],
            src.QUALITY_PROTOCOL_CONTRACT_SHA256,
        )
        self.assertEqual(
            src.QUALITY_PROTOCOL_CONTRACT_SHA256,
            _independent_sha256(src.QUALITY_PROTOCOL_CONTRACT),
        )
        contract = deps["quality_protocol"]["contract"]
        self.assertEqual(
            tuple(contract["identity_fields"]), src.QUALITY_ACK_IDENTITY_FIELDS
        )
        self.assertEqual(
            tuple(contract["quality_fields"]), src.QUALITY_ACK_QUALITY_FIELDS
        )
        self.assertEqual(
            tuple(contract["timing_fields"]), src.QUALITY_ACK_TIMING_FIELDS
        )

        # and the declared literals still match the real module exactly
        self.assertIs(
            src.verify_quality_protocol_binding(), src.QUALITY_PROTOCOL_CONTRACT
        )
        for name, expected in (
            ("QUALITY_EVALUATED_ACK_SCHEMA", src.QUALITY_ACK_SCHEMA),
            ("QUALITY_EVALUATION_FAILED_ACK_SCHEMA", src.QUALITY_ACK_FAILURE_SCHEMA),
            ("PROTOCOL_VERSION", src.QUALITY_ACK_PROTOCOL_VERSION),
            ("SOURCE", src.QUALITY_ACK_SOURCE),
        ):
            with self.subTest(constant=name):
                self.assertEqual(getattr(self.protocol, name), expected)
        for name, expected in (
            ("IDENTITY_FIELDS", src.QUALITY_ACK_IDENTITY_FIELDS),
            ("TIMING_FIELDS", src.QUALITY_ACK_TIMING_FIELDS),
            ("QUALITY_FIELDS", src.QUALITY_ACK_QUALITY_FIELDS),
        ):
            with self.subTest(constant=name):
                self.assertEqual(tuple(getattr(self.protocol, name)), expected)
        # no false-positive field exists anywhere in the score layout
        for name in self.protocol.QUALITY_FIELDS:
            with self.subTest(field=name):
                self.assertFalse(name.endswith("_fp"))
                self.assertNotIn("false_positive", name)
                self.assertNotIn("precision", name)
        self.assertFalse(src.QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE)
        self.assertEqual(src.QUALITY_ACK_DERIVABLE_DETECTION_METRICS, ("recall",))

        # the deployability and scope statements are explicit
        self.assertEqual(
            descriptor["causal_state"]["deployability"], "SIMULATOR_TESTBED_ONLY"
        )
        self.assertFalse(descriptor["quality"]["gt_deployable"])
        self.assertEqual(
            descriptor["eligibility"]["implemented_scope"], "PER_UE_PERCEPTION"
        )
        self.assertIn("never summed", descriptor["eligibility"]["scope_separation"])
        with self.assertRaises(TypeError):
            descriptor["version"] = 3  # type: ignore[index]

    # ------------------------------------------------------------------ 3 -- #

    def test_quality_is_localization_based_and_monotone(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        eligibility = self._eligibility()

        def _evaluate(**quality_overrides: Any) -> src.QualityEvaluationV1:
            document = self._ack_document(
                ticket.action,
                frame_id=ticket.reward_carla_frame_id,
                quality=self._ack_quality(**quality_overrides),
            )
            evidence = self._evidence(
                ticket,
                obligation=obligation,
                binding=self._binding(
                    ticket, document=document, obligation=obligation
                ),
                eligibility=eligibility,
            )
            return spec.evaluate_quality(self._components(evidence=evidence))

        base = _evaluate()
        self.assertTrue(base.is_attested)

        # (a) lower XY error strictly improves Q
        errors = [0.0, 0.25, 0.95, 2.0, 8.0]
        qualities = [
            _evaluate(
                vehicle={"source_time_world_xy_error_m": e},
                person={"source_time_world_xy_error_m": e},
            ).q_perc
            for e in errors
        ]
        for better, worse in zip(qualities, qualities[1:]):
            self.assertGreater(better, worse)

        # (b) higher recall strictly improves Q
        recalls = []
        for tp, fn in ((4, 0), (3, 1), (2, 2), (1, 3)):
            evaluation = _evaluate(vehicle={"tp": tp, "fn": fn})
            recalls.append(evaluation.q_perc)
            self.assertAlmostEqual(
                evaluation.per_class_recall["vehicle"], tp / 4.0, 12
            )
        for better, worse in zip(recalls, recalls[1:]):
            self.assertGreater(better, worse)

        # (c) U_loc = sqrt(recall * exp(-e/tau)), recomputed independently
        checked = _evaluate(
            vehicle={"tp": 3, "fn": 1, "source_time_world_xy_error_m": 0.95}
        )
        self.assertAlmostEqual(
            checked.per_class_localization_utility["vehicle"],
            math.sqrt(0.75 * math.exp(-0.95 / spec.tau_vehicle_m)),
            12,
        )

        # (d) a complete miss of eligible objects yields zero class utility
        missed = _evaluate(
            person={"tp": 0, "fn": 2, "recall": 0.0,
                    "source_time_world_xy_error_m": None}
        )
        self.assertEqual(missed.per_class_localization_utility["person"], 0.0)
        self.assertEqual(missed.per_class_recall["person"], 0.0)
        self.assertIn("person", missed.localization_weights_used)
        self.assertIn(
            "person", missed.components.missed_localization_classes
        )
        # under the geometric combiner a missed eligible class collapses Q_loc
        self.assertEqual(missed.q_loc, 0.0)
        self.assertEqual(missed.q_perc, 0.0)

        # (e) strong segmentation cannot rescue collapsed localization
        rescued = _evaluate(
            person={"tp": 0, "fn": 2, "recall": 0.0,
                    "source_time_world_xy_error_m": None},
            segmentation={"miou_vehicle_iou": 1.0, "miou_person_iou": 1.0},
        )
        self.assertEqual(rescued.q_seg, 1.0)
        self.assertEqual(rescued.q_perc, 0.0)

        # (f) the frozen references map to normalized 1
        at_reference = _evaluate(
            segmentation={
                "miou_vehicle_iou": spec.seg_reference_vehicle_iou,
                "miou_person_iou": spec.seg_reference_person_iou,
            }
        )
        self.assertAlmostEqual(
            at_reference.per_class_normalized_segmentation["vehicle"], 1.0, 12
        )
        self.assertAlmostEqual(
            at_reference.per_class_normalized_segmentation["person"], 1.0, 12
        )
        self.assertAlmostEqual(at_reference.q_seg, 1.0, 12)
        # and exceeding the reference clips rather than exceeding 1
        above = _evaluate(segmentation={"miou_vehicle_iou": 1.0})
        self.assertAlmostEqual(
            above.per_class_normalized_segmentation["vehicle"], 1.0, 12
        )

        # (g) Q_perc = Q_loc * ((1 - beta) + beta * Q_seg), recomputed
        beta = float(spec.segmentation_modulation_beta)
        self.assertAlmostEqual(
            base.q_perc,
            base.q_loc * ((1.0 - beta) + beta * base.q_seg),
            12,
        )
        # the arithmetic combiner keeps a missed class in without collapsing
        arithmetic = self._reward_spec(
            localization_combiner=(
                src.LocalizationCombiner.WEIGHTED_ARITHMETIC_MEAN
            )
        )
        document = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                person={"tp": 0, "fn": 2, "recall": 0.0,
                        "source_time_world_xy_error_m": None}
            ),
        )
        evidence = self._evidence(
            ticket,
            obligation=obligation,
            binding=self._binding(
                ticket, document=document, obligation=obligation
            ),
            eligibility=eligibility,
        )
        mixed = arithmetic.evaluate_quality(self._components(evidence=evidence))
        self.assertEqual(mixed.per_class_localization_utility["person"], 0.0)
        self.assertGreater(mixed.q_loc, 0.0)
        self.assertIn("person", mixed.localization_weights_used)

    # ------------------------------------------------------------------ 4 -- #

    def test_segmentation_exclusion_only_when_both_masks_empty(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        obligation = self._obligation(ticket)

        def _components(
            *, seg_overrides: Mapping[str, Any], pred_v: int, pred_p: int
        ) -> src.QualityComponentsV1:
            document = self._ack_document(
                ticket.action,
                frame_id=ticket.reward_carla_frame_id,
                quality=self._ack_quality(segmentation=dict(seg_overrides)),
            )
            evidence = self._evidence(
                ticket,
                obligation=obligation,
                binding=self._binding(
                    ticket, document=document, obligation=obligation
                ),
            )
            return self._components(
                evidence=evidence,
                pred_vehicle_pixels=pred_v,
                pred_person_pixels=pred_p,
            )

        # GT absent + predicted false-positive mask + IoU 0 stays VALID
        false_positive = _components(
            seg_overrides={"gt_person_pixels": 0, "miou_person_iou": 0.0},
            pred_v=1100,
            pred_p=450,
        )
        self.assertTrue(false_positive.person_segmentation.is_defined)
        self.assertIn("person", false_positive.defined_segmentation_classes)
        penalized = spec.evaluate_quality(false_positive)
        self.assertIn("person", penalized.segmentation_weights_used)
        self.assertEqual(
            penalized.per_class_normalized_segmentation["person"], 0.0
        )
        # a zero normalized class collapses the geometric segmentation mean
        self.assertEqual(penalized.q_seg, 0.0)
        self.assertLess(penalized.q_perc, penalized.q_loc)

        # GT present + missed mask + IoU 0 stays VALID and is penalized
        missed_mask = _components(
            seg_overrides={"miou_person_iou": 0.0},
            pred_v=1100,
            pred_p=0,
        )
        self.assertTrue(missed_mask.person_segmentation.is_defined)
        self.assertEqual(
            spec.evaluate_quality(missed_mask).q_seg, 0.0
        )

        # only the genuinely vacuous case drops out
        vacuous = _components(
            seg_overrides={"gt_person_pixels": 0, "miou_person_iou": 0.0},
            pred_v=1100,
            pred_p=0,
        )
        self.assertFalse(vacuous.person_segmentation.is_defined)
        self.assertEqual(
            vacuous.person_segmentation.exclusion_reason, "both_masks_empty"
        )
        evaluated = spec.evaluate_quality(vacuous)
        self.assertNotIn("person", evaluated.segmentation_weights_used)
        self.assertIn("vehicle", evaluated.segmentation_weights_used)

    # ------------------------------------------------------------------ 5 -- #

    def test_reward_scalar_includes_switch_penalties(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        components = self._components(ticket)
        previous = self._action(mode_id=5, q_e4=5000)

        outcome = src.evaluate_completed_decision(
            ticket,
            spec,
            quality_components=components,
            previous_action=previous,
        )
        self.assertTrue(outcome.is_attested)
        assert outcome.quality is not None and outcome.latency is not None
        switch = outcome.switch_penalty
        self.assertTrue(switch.applicable)
        self.assertTrue(switch.mode_changed)
        self.assertAlmostEqual(switch.q_exec_delta, abs(0.98 - 0.50), 12)
        self.assertAlmostEqual(
            switch.total, spec.lambda_mode + spec.lambda_q * 0.48, 12
        )
        # r = w_Q Q - w_L (L/B) - lambda_m 1[mode changed] - lambda_q |dq|
        self.assertAlmostEqual(
            outcome.scalar_reward,
            spec.w_quality * outcome.quality.q_perc
            - spec.w_latency * outcome.latency.normalized_latency
            - switch.total,
            12,
        )
        self.assertEqual(outcome.latency.l_ns, 50 * MS)
        self.assertEqual(outcome.latency.clock_domain, "UE_LOCAL_MONOTONIC")

        # same mode, same q: no penalty at all
        same = src.evaluate_completed_decision(
            ticket,
            spec,
            quality_components=components,
            previous_action=ticket.action,
        )
        self.assertFalse(same.switch_penalty.mode_changed)
        self.assertEqual(same.switch_penalty.q_exec_delta, 0.0)
        self.assertEqual(same.switch_penalty.total, 0.0)

        # episode start: inapplicable, explicitly, not an indistinguishable zero
        first = src.evaluate_completed_decision(
            ticket, spec, quality_components=components, previous_action=None
        )
        self.assertFalse(first.switch_penalty.applicable)
        self.assertIsNone(first.switch_penalty.mode_changed)
        self.assertIsNone(first.switch_penalty.q_exec_delta)
        self.assertEqual(first.switch_penalty.total, 0.0)

        # the latency-excess diagnostic is outside the optimization contract
        self.assertEqual(
            src.OPTIMIZATION_CONTRACT_COSTS,
            ("c_deadline", "c_authoritative_failure"),
        )
        self.assertEqual(src.DIAGNOSTIC_ONLY_SIGNALS, ("c_latency_excess",))
        self.assertNotIn("c_latency_excess", outcome.costs.to_canonical_dict())
        self.assertFalse(outcome.diagnostics.informative)
        self.assertEqual(outcome.diagnostics.c_latency_excess, 0.0)
        with self.assertRaises(src.StateRewardContractError):
            src.DiagnosticSignalsV1(c_latency_excess=0.0, informative=True)

    # ------------------------------------------------------------------ 6 -- #

    def test_state_is_typed_timestamped_and_ages_are_derived(self) -> None:
        state = self._state()
        norm, fresh = self._norm(), self._freshness()

        # ages are derived, never supplied
        ages = state.measurement_ages_ns
        self.assertEqual(ages["scene"], 20 * MS)
        self.assertEqual(ages["snr"], 30 * MS)
        self.assertEqual(ages["bsr"], 25 * MS)
        self.assertEqual(ages["mcs"], 35 * MS)
        self.assertEqual(
            state.scene.age_ns(state.observed_ns),
            state.observed_ns - state.scene.measured_ns,
        )
        # the scene descriptor is bound to its frame and source hash
        self.assertEqual(state.scene.carla_frame_id, state.carla_frame_id)
        self.assertEqual(state.scene.source_sha256, HEX_B)
        # radio telemetry is typed
        self.assertIs(
            state.radio.snr_metric,
            src.SnrMetric.UL_PUSCH_POST_EQUALISER_SINR_DB,
        )
        self.assertIs(state.radio.snr_direction, src.LinkDirection.UPLINK)
        self.assertIs(state.radio.bsr_scope, src.BsrScope.ALL_GROUPS_LATEST)
        self.assertEqual(state.radio.bsr_logical_channel_group, 1)
        self.assertIs(state.clock_domain, src.ClockDomain.UE_LOCAL_MONOTONIC)
        self.assertEqual(len(src.ClockDomain), 1)

        vector = src.build_policy_features(state, norm, fresh)
        self.assertEqual(len(vector.as_tuple()), src.POLICY_FEATURE_COUNT)
        self.assertEqual(src.POLICY_FEATURE_COUNT, 31)
        named = vector.as_mapping()
        # normalized ages use the registered bounds as denominators
        self.assertAlmostEqual(
            named["freshness_scene_normalized"], (20 * MS) / (150 * MS), 12
        )
        self.assertAlmostEqual(
            named["freshness_snr_normalized"], (30 * MS) / (200 * MS), 12
        )
        # the complete freshness-policy hash is bound, not just the id
        self.assertEqual(
            vector.freshness_policy_sha256, fresh.canonical_sha256()
        )
        self.assertEqual(vector.freshness_policy_id, fresh.policy_id)
        # and the observation schema declares itself simulator-only
        self.assertEqual(vector.deployability, "SIMULATOR_TESTBED_ONLY")
        self.assertEqual(
            vector.to_canonical_dict()["deployability"],
            "SIMULATOR_TESTBED_ONLY",
        )

        # identifiers never enter the vector
        src.assert_policy_features_exclude_forbidden_fields()
        for name in src.POLICY_FEATURE_ORDER:
            for forbidden in src.FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
                self.assertNotIn(forbidden, name)
        relabelled = self._state(
            session_uuid=OTHER_SESSION, tensor_seq=987, observed_ns=T0
        )
        self.assertEqual(
            src.build_policy_features(relabelled, norm, fresh).as_tuple(),
            vector.as_tuple(),
        )

        # a previous decision fills exactly one mode slot and one terminal slot
        ticket = self._ticket()
        spec = self._reward_spec()
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._components(ticket)
        )
        previous = src.PreviousOutcomeV1.from_completed(ticket, outcome, spec)
        self.assertTrue(previous.is_attested)
        with_prev = self._state(previous=previous)
        prev_named = src.build_policy_features(
            with_prev, norm, fresh
        ).as_mapping()
        self.assertEqual(prev_named["prev_present_mask"], 1.0)
        self.assertEqual(prev_named["prev_terminal_onehot_exact"], 1.0)
        self.assertEqual(
            prev_named["prev_terminal_onehot_feedback_timeout"], 0.0
        )
        self.assertEqual(
            sum(
                prev_named[f"prev_terminal_onehot_{code}"]
                for code in src.PREVIOUS_TERMINAL_FEATURE_CODES.values()
            ),
            1.0,
        )
        self.assertEqual(prev_named["prev_quality_valid_mask"], 1.0)
        assert outcome.quality is not None
        self.assertAlmostEqual(
            prev_named["prev_quality_normalized"], outcome.quality.q_perc, 12
        )
        # the previous outcome is bound to its ticket, outcome and spec
        self.assertEqual(
            previous.completed_ticket_sha256, ticket.canonical_sha256()
        )
        self.assertEqual(previous.outcome_sha256, outcome.canonical_sha256())
        self.assertEqual(previous.reward_spec_sha256, spec.canonical_sha256())
        self.assertEqual(previous.resolution_ns, ticket.resolution_ns)
        self.assertEqual(previous.quality_gt_source, "CARLA_GT_EXACT")
        self.assertEqual(previous.latency_clock_domain, "UE_LOCAL_MONOTONIC")

        # a timeout and an action-path failure are distinguishable observations
        codes = {}
        for terminal in (
            rtc.TerminalClass.ACTION_PATH_FAILURE,
            rtc.TerminalClass.FEEDBACK_TIMEOUT,
        ):
            other = self._ticket(terminal=terminal)
            other_outcome = src.evaluate_completed_decision(other, spec)
            other_prev = src.PreviousOutcomeV1.from_completed(
                other, other_outcome, spec
            )
            named_other = src.build_policy_features(
                self._state(previous=other_prev), norm, fresh
            ).as_mapping()
            code = src.PREVIOUS_TERMINAL_FEATURE_CODES[terminal]
            self.assertEqual(named_other[f"prev_terminal_onehot_{code}"], 1.0)
            codes[terminal] = tuple(
                named_other[f"prev_terminal_onehot_{c}"]
                for c in src.PREVIOUS_TERMINAL_FEATURE_CODES.values()
            )
        self.assertNotEqual(
            codes[rtc.TerminalClass.ACTION_PATH_FAILURE],
            codes[rtc.TerminalClass.FEEDBACK_TIMEOUT],
        )

    # ------------------------------------------------------------------ 7 -- #

    def test_transition_derives_duration_discount_and_revalidates(self) -> None:
        for extra_reuses, expected_d in ((0, 2), (2, 4)):
            with self.subTest(d=expected_d):
                spec = self._reward_spec(gamma_per_tensor=0.97)
                transition = self._full_transition(
                    spec=spec, extra_reuses=extra_reuses
                )
                self.assertEqual(transition.hold_duration_tensors, expected_d)
                self.assertEqual(
                    transition.hold_duration_tensors,
                    transition.hold.tensor_count,
                )
                self.assertAlmostEqual(
                    transition.discount_multiplier, 0.97 ** expected_d, 12
                )
                self.assertGreaterEqual(expected_d, rtc.K_MIN_TENSORS)
                # re-derivation from frozen sources reproduces the outcome
                self.assertEqual(
                    transition.revalidate().canonical_sha256(),
                    transition.outcome.canonical_sha256(),
                )

        import inspect

        parameters = set(
            inspect.signature(src.ReplayTransitionV1.__init__).parameters
        )
        for forbidden in (
            "hold_duration_tensors", "discount_multiplier", "d", "duration",
            "gamma_per_tensor",
        ):
            with self.subTest(parameter=forbidden):
                self.assertNotIn(forbidden, parameters)

        # the policy trace records the sample and verifies the quantization
        transition = self._full_transition()
        trace = transition.policy_trace
        self.assertEqual(trace.sampled_mode_id, transition.executed_action.mode_id)
        self.assertEqual(trace.q_exec, transition.executed_action.q_e4 / 1e4)
        self.assertLessEqual(trace.log_prob_discrete, 0.0)
        self.assertLessEqual(trace.log_prob_continuous, 0.0)
        self.assertEqual(trace.actor_version_sha256, HEX_B)
        # an unrounded sample still quantizes correctly under half-up
        self._trace(
            self._action(mode_id=3, q_e4=9800), sampled_q=0.97996
        )

        # both evidence hashes are bound into the transition
        self.assertIsNotNone(transition.raw_quality_ack_sha256)
        self.assertIsNotNone(transition.detailed_evidence_sha256)
        payload = transition.to_canonical_dict()
        for key in (
            "raw_quality_ack_sha256", "detailed_evidence_sha256",
            "executed_action_sha256", "policy_trace", "freshness_policy_sha256",
            "state_normalization_spec_sha256", "reward_latency_clock_domain",
            "hold_duration_tensors", "discount_multiplier", "next_state",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(
            transition.canonical_sha256(), _independent_sha256(payload)
        )
        self.assertEqual(
            transition.canonical_bytes(), _independent_canonical_bytes(payload)
        )
        # next_state.previous is exactly this completed decision
        assert transition.next_state is not None
        successor = transition.next_state.previous
        assert successor is not None
        self.assertEqual(
            successor.completed_ticket_sha256,
            transition.completed_ticket.canonical_sha256(),
        )
        self.assertEqual(successor.decision_seq, transition.decision_seq)

    # ------------------------------------------------------------------ 8 -- #

    def test_terminal_classes_and_eligibility(self) -> None:
        spec = self._reward_spec()

        # action-path failure is a registered negative, no fabricated Q
        failure = self._ticket(terminal=rtc.TerminalClass.ACTION_PATH_FAILURE)
        failed = src.evaluate_completed_decision(failure, spec)
        self.assertIs(failed.eligibility, src.LearningEligibility.ELIGIBLE)
        self.assertIsNone(failed.quality)
        self.assertEqual(failed.costs.c_authoritative_failure, 1.0)
        self.assertAlmostEqual(failed.scalar_reward, -1.0, 12)

        # timeout is censored until an adjudication bound to this ticket
        timeout = self._ticket(terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT)
        censored = src.evaluate_completed_decision(timeout, spec)
        self.assertIs(
            censored.eligibility,
            src.LearningEligibility.CENSORED_PENDING_ADJUDICATION,
        )
        self.assertIsNone(censored.scalar_reward)
        # a proven feedback-only loss is never punished
        loss = src.evaluate_completed_decision(
            timeout,
            spec,
            adjudication=self._adjudication(
                timeout, src.Adjudication.FEEDBACK_ONLY_LOSS
            ),
        )
        self.assertIs(
            loss.eligibility,
            src.LearningEligibility.CENSORED_FEEDBACK_ONLY_LOSS,
        )
        self.assertIsNone(loss.scalar_reward)
        # an authoritative service failure becomes the registered negative
        adjudicated = src.evaluate_completed_decision(
            timeout,
            spec,
            adjudication=self._adjudication(
                timeout, src.Adjudication.AUTHORITATIVE_SERVICE_FAILURE
            ),
        )
        self.assertIs(adjudicated.eligibility, src.LearningEligibility.ELIGIBLE)
        self.assertAlmostEqual(adjudicated.scalar_reward, -1.0, 12)
        self.assertEqual(adjudicated.costs.c_authoritative_failure, 1.0)
        # an adjudicated instrument fault is excluded
        excluded = src.evaluate_completed_decision(
            timeout,
            spec,
            adjudication=self._adjudication(
                timeout, src.Adjudication.INFRASTRUCTURE_FAULT
            ),
        )
        self.assertIs(
            excluded.eligibility,
            src.LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
        )
        self.assertIsNone(excluded.scalar_reward)

        # a controller infrastructure fault is excluded and never a penalty
        fault = self._ticket(
            terminal=rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED
        )
        faulted = src.evaluate_completed_decision(fault, spec)
        self.assertIs(
            faulted.eligibility,
            src.LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
        )
        self.assertIsNone(faulted.scalar_reward)
        self.assertIsNone(faulted.costs.c_authoritative_failure)
        self.assertNotEqual(faulted.scalar_reward, spec.r_registered_failure)
        self.assertEqual(
            rtc.TERMINAL_LEARNING_DISPOSITION[fault.terminal_class],
            "excluded_reported_as_experimental_failure",
        )

    # ------------------------------------------------------------------ 9 -- #

    def test_no_import_side_effects(self) -> None:
        """Importing the contract reads no file and opens no socket."""
        probe = r'''
import builtins, io, sys
import numpy, cv2  # third-party init first

_hits = []


def _blocked(*args, **kwargs):
    _hits.append(args[:1])
    raise AssertionError(f"filesystem access during import: {args[:1]}")


builtins.open = _blocked
io.open = _blocked

import socket


def _blocked_socket(*args, **kwargs):
    raise AssertionError("network access during import")


socket.socket = _blocked_socket
socket.create_connection = _blocked_socket

from abiodun.rl_agent.splitfusion_hybrid_sac_v1 import (  # noqa: F401
    state_reward_transition_contract as contract,
)
from abiodun.rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac

assert ac.default_contract.cache_info().currsize == 0, "catalog read on import"
# the quality-protocol module must NOT have been loaded eagerly
assert (
    "rl_agent.splitfusion_quality_feedback_probe_v1.protocol"
    not in sys.modules
), "quality protocol imported eagerly"
for banned in ("carla", "torch", "docker", "pycuda", "tensorflow"):
    assert banned not in sys.modules, banned
assert not _hits, _hits
assert contract.SCHEMA_VERSION == 2
print("IMPORT_CLEAN", contract.SCHEMA_SHA256)
'''
        package_root = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(package_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"probe failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
        self.assertIn("IMPORT_CLEAN", completed.stdout)
        self.assertIn(src.SCHEMA_SHA256, completed.stdout)

        for banned in (
            "Env", "GymEnv", "ReplayBuffer", "Actor", "Critic", "train",
            "optimizer", "LSTM", "GRU", "step", "reset",
        ):
            with self.subTest(symbol=banned):
                self.assertNotIn(banned, src.__all__)

    # ----------------------------------------------------------------- 10 -- #

    def test_no_eligible_ground_truth_is_censored_not_scored_zero(self) -> None:
        """The extreme of the eligibility rule: nothing was this UE's to find."""
        spec = self._reward_spec()
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        document = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                vehicle={"tp": 0, "fn": 0, "recall": None,
                         "source_time_world_xy_error_m": None},
                person={"tp": 0, "fn": 0, "recall": None,
                        "source_time_world_xy_error_m": None},
            ),
        )
        evidence = self._evidence(
            ticket,
            obligation=obligation,
            binding=self._binding(
                ticket, document=document, obligation=obligation
            ),
        )
        components = self._components(
            evidence=evidence,
            vehicle_eligible_gt_instances=0,
            person_eligible_gt_instances=0,
        )
        self.assertEqual(
            components.undefined_localization_classes, ("vehicle", "person")
        )
        self.assertEqual(components.defined_localization_classes, ())

        # Q_loc is undefined, so the quality itself refuses to exist ...
        with self.assertRaises(src.InsufficientQualitySupportError) as caught:
            spec.evaluate_quality(components)
        self.assertIn("no per-UE perception reward", str(caught.exception))

        # ... and the decision is CENSORED, not scored zero.  A frame with
        # nothing eligible to perceive is not a failure of the action.
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=components
        )
        self.assertIs(
            outcome.eligibility,
            src.LearningEligibility.CENSORED_NO_ELIGIBLE_GROUND_TRUTH,
        )
        self.assertIsNone(outcome.scalar_reward)
        self.assertIsNone(outcome.quality)
        self.assertFalse(outcome.learning_eligible)
        # the latency is still a real measurement and is preserved
        assert outcome.latency is not None
        self.assertEqual(outcome.latency.l_ns, 50 * MS)
        self.assertEqual(outcome.costs.c_authoritative_failure, 0.0)
        self.assertNotEqual(outcome.scalar_reward, 0.0)

        # a transition over it carries no reward and re-derives cleanly
        transition = src.build_replay_transition(
            state=self._state(),
            next_state=self._next_state(ticket, outcome, spec),
            completed_ticket=ticket,
            outcome=outcome,
            policy_trace=self._trace(ticket.action),
            reward_spec=spec,
            normalization=self._norm(),
            freshness=self._freshness(),
        )
        self.assertIsNone(transition.scalar_reward)
        self.assertIsNone(transition.quality)
        # the ACK was genuinely received and verified, so its evidence hashes
        # stay bound even though the frame earned no reward -- the censoring is
        # auditable rather than silent
        self.assertIsNotNone(transition.quality_evidence)
        self.assertIsNotNone(transition.raw_quality_ack_sha256)
        self.assertIsNotNone(transition.detailed_evidence_sha256)
        self.assertIsNotNone(transition.quality_components)
        self.assertEqual(
            transition.revalidate().canonical_sha256(),
            outcome.canonical_sha256(),
        )

        # when every segmentation class is vacuous too, the modulation is the
        # identity rather than a punitive zero: there is no segmentation
        # evidence to modulate with.
        vacuous_seg = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                segmentation={
                    "gt_vehicle_pixels": 0, "gt_person_pixels": 0,
                    "miou_vehicle_iou": 0.0, "miou_person_iou": 0.0,
                }
            ),
        )
        seg_evidence = self._evidence(
            ticket,
            obligation=obligation,
            binding=self._binding(
                ticket, document=vacuous_seg, obligation=obligation
            ),
        )
        seg_components = self._components(
            evidence=seg_evidence, pred_vehicle_pixels=0, pred_person_pixels=0
        )
        self.assertEqual(seg_components.defined_segmentation_classes, ())
        evaluated = spec.evaluate_quality(seg_components)
        self.assertIsNone(evaluated.q_seg)
        self.assertEqual(evaluated.q_perc, evaluated.q_loc)
        self.assertGreater(evaluated.q_perc, 0.0)

    # ----------------------------------------------------------------- 11 -- #

    def test_serialization_sweep_and_terminal_transitions(self) -> None:
        spec, norm, fresh = self._reward_spec(), self._norm(), self._freshness()
        ticket = self._ticket()
        components = self._components(ticket)
        # the switch penalty is derived from the decision the state carries,
        # so the two must be built together
        earlier = self._ticket(
            decision_seq=0, first_tensor_seq=4, first_frame_id=400,
            action=self._action(mode_id=5, q_e4=5000),
        )
        earlier_outcome = src.evaluate_completed_decision(
            earlier, spec, quality_components=self._components(earlier)
        )
        state = self._state(
            previous=src.PreviousOutcomeV1.from_completed(
                earlier, earlier_outcome, spec
            )
        )
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=components,
            previous_action=earlier.action,
        )
        evaluation = outcome.quality
        assert evaluation is not None
        latency = outcome.latency
        assert latency is not None

        for label, record in (
            ("eligibility", self._eligibility()),
            ("obligation", self._obligation(ticket)),
            ("ack_binding", self._binding(ticket)),
            ("evidence", components.evidence),
            ("components", components),
            ("evaluation", evaluation),
            ("latency", latency),
            ("reward_spec", spec),
            ("normalization", norm),
            ("freshness", fresh),
            ("state", state),
            ("feature_vector", src.build_policy_features(state, norm, fresh)),
            ("switch_penalty", outcome.switch_penalty),
            ("costs", outcome.costs),
            ("diagnostics", outcome.diagnostics),
            ("outcome", outcome),
            ("policy_trace", self._trace(ticket.action)),
            ("previous", src.PreviousOutcomeV1.from_completed(
                ticket, outcome, spec
            )),
            ("adjudication", self._adjudication(
                self._ticket(terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT),
                src.Adjudication.PENDING,
            )),
        ):
            with self.subTest(record=label):
                payload = record.to_canonical_dict()
                self.assertIn("record", payload)
                raw = _independent_canonical_bytes(payload)
                self.assertEqual(raw, _independent_canonical_bytes(payload))
                if hasattr(record, "canonical_sha256"):
                    self.assertEqual(
                        record.canonical_sha256(), _independent_sha256(payload)
                    )
                if hasattr(record, "canonical_bytes"):
                    self.assertEqual(record.canonical_bytes(), raw)

        # a few derived accessors
        self.assertEqual(evaluation.quality, evaluation.q_perc)
        self.assertIs(components.gt_source, src.GroundTruthSource.CARLA_GT_EXACT)
        self.assertTrue(state.has_previous_decision)
        self.assertFalse(self._state().has_previous_decision)
        self.assertEqual(
            src.build_policy_features(state, norm, fresh).feature_names,
            src.POLICY_FEATURE_ORDER,
        )
        self.assertAlmostEqual(
            spec.normalize_localization(1.0, 2.0), math.exp(-0.5), 12
        )

        # a terminated transition may omit the next state; a non-terminal one
        # may not
        terminated = src.build_replay_transition(
            state=state,
            next_state=None,
            completed_ticket=ticket,
            outcome=outcome,
            policy_trace=self._trace(ticket.action),
            reward_spec=spec,
            normalization=norm,
            freshness=fresh,
            terminated=True,
            episode_end_reason="route complete",
        )
        self.assertTrue(terminated.terminated)
        self.assertIsNone(terminated.next_state)
        self.assertIsNone(terminated.to_canonical_dict()["next_state"])
        truncated = src.build_replay_transition(
            state=state,
            next_state=None,
            completed_ticket=ticket,
            outcome=outcome,
            policy_trace=self._trace(ticket.action),
            reward_spec=spec,
            normalization=norm,
            freshness=fresh,
            truncated=True,
            episode_end_reason="time limit",
        )
        self.assertTrue(truncated.truncated)
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=state,
                next_state=None,
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=norm,
                freshness=fresh,
            )
        self.assertIn("requires a next state", str(caught.exception))
        for kwargs in (
            {"terminated": True, "truncated": True,
             "episode_end_reason": "both"},
            {"terminated": True},
            {"episode_end_reason": "reason without an ending"},
        ):
            with self.subTest(kwargs=sorted(kwargs)):
                with self.assertRaises(src.TransitionIdentityError):
                    src.build_replay_transition(
                        state=state,
                        next_state=None,
                        completed_ticket=ticket,
                        outcome=outcome,
                        policy_trace=self._trace(ticket.action),
                        reward_spec=spec,
                        normalization=norm,
                        freshness=fresh,
                        **kwargs,
                    )


class AdversarialRejectionTest(BaseContractTest):
    """The fourteen required rejection proofs, each named for its attack."""

    # ------------------------------------------------------------------ 1 -- #

    def test_reject_ack_reused_for_another_frame_or_decision(self) -> None:
        ticket = self._ticket(decision_seq=1, first_frame_id=500)
        obligation = self._obligation(ticket)

        # an ACK describing a different frame cannot bind
        wrong_frame = self._ack_document(ticket.action, frame_id=501)
        with self.assertRaises(src.QualityContractError) as caught:
            src.QualityAckBindingV1.from_ack_document(
                wrong_frame, obligation=obligation, completed_ticket=ticket
            )
        self.assertIn("frame", str(caught.exception))

        # a valid ACK for decision 1 cannot be replayed onto decision 2
        document = self._ack_document(ticket.action, frame_id=500)
        other_ticket = self._ticket(
            decision_seq=2, first_tensor_seq=20, first_frame_id=600
        )
        with self.assertRaises(src.QualityContractError):
            src.QualityAckBindingV1.from_ack_document(
                document,
                obligation=self._obligation(other_ticket),
                completed_ticket=other_ticket,
            )
        # nor onto a mismatched obligation/ticket pair
        with self.assertRaises(src.QualityContractError):
            src.QualityAckBindingV1.from_ack_document(
                document, obligation=obligation, completed_ticket=other_ticket
            )
        # a different run/cell/stream is refused too
        for field in ("run_id", "cell_id", "stream_id"):
            with self.subTest(field=field):
                with self.assertRaises(src.QualityContractError):
                    src.QualityAckBindingV1.from_ack_document(
                        self._ack_document(
                            ticket.action, frame_id=500, **{field: "other"}
                        ),
                        obligation=obligation,
                        completed_ticket=ticket,
                    )
        # a different capture timestamp is refused
        with self.assertRaises(src.QualityContractError):
            src.QualityAckBindingV1.from_ack_document(
                self._ack_document(
                    ticket.action, frame_id=500,
                    capture_timestamp_ns=WALL0 + 1,
                ),
                obligation=obligation,
                completed_ticket=ticket,
            )
        # and an ACK naming a different anchor action is refused
        with self.assertRaises(src.QualityContractError):
            src.QualityAckBindingV1.from_ack_document(
                self._ack_document(self._action(mode_id=5, q_e4=5000),
                                   frame_id=500),
                obligation=obligation,
                completed_ticket=ticket,
            )

    # ------------------------------------------------------------------ 2 -- #

    def test_reject_caller_hash_not_matching_the_document(self) -> None:
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        document = self._ack_document(
            ticket.action, frame_id=ticket.reward_carla_frame_id
        )
        with self.assertRaises(src.QualityContractError) as caught:
            src.QualityAckBindingV1.from_ack_document(
                document,
                obligation=obligation,
                completed_ticket=ticket,
                expected_raw_ack_sha256=HEX_A,
            )
        self.assertIn("recomputed", str(caught.exception))

        # an opaque hash is never accepted in place of the document at all
        for impostor in (HEX_A, {"dh": HEX_B}, None, 42):
            with self.subTest(document=type(impostor).__name__):
                with self.assertRaises(src.QualityContractError):
                    src.QualityAckBindingV1.from_ack_document(
                        impostor, obligation=obligation, completed_ticket=ticket
                    )
        # a tampered document fails the real validator, not a local check
        tampered = dict(document)
        tampered["pg"] = False
        with self.assertRaises(src.QualityContractError) as caught:
            src.QualityAckBindingV1.from_ack_document(
                tampered, obligation=obligation, completed_ticket=ticket
            )
        self.assertIn("validator", str(caught.exception))
        # a failure ACK carries no scores and cannot be bound
        failed = dict(document)
        failed["s"] = src.QUALITY_ACK_FAILURE_SCHEMA
        failed["q"] = []
        failed["r"] = "evaluator crashed"
        with self.assertRaises(src.QualityContractError):
            src.QualityAckBindingV1.from_ack_document(
                failed, obligation=obligation, completed_ticket=ticket
            )
        # a forged binding cannot be constructed directly and serialized
        real = self._binding(ticket)
        forged = src.QualityAckBindingV1(
            raw_quality_ack_sha256=HEX_A,
            detailed_evidence_sha256=HEX_B,
            ack_schema=src.QUALITY_ACK_SCHEMA,
            ack_protocol_version=1,
            ack_source=src.QUALITY_ACK_SOURCE,
            identity_fields=dict(real.identity_fields),
            quality_fields=dict(real.quality_fields),
            evaluator_mode="forged",
            obligation_sha256=obligation.canonical_sha256(),
            completed_ticket_sha256=ticket.canonical_sha256(),
        )
        self.assertFalse(forged.is_attested)
        with self.assertRaises(src.UnattestedRecordError):
            forged.to_canonical_dict()

    # ------------------------------------------------------------------ 3 -- #

    def test_reject_off_anchor_eligible_quality_with_null_evidence(self) -> None:
        off_anchor = self._off_anchor_action()
        ticket = self._ticket(action=off_anchor)
        obligation = self._obligation(ticket)
        self.assertFalse(obligation.is_anchor_expressible)

        # the v1 ACK cannot name it, and nothing is snapped
        with self.assertRaises(src.OffAnchorQualityAckError) as caught:
            src.QualityAckBindingV1.from_ack_document(
                self._ack_document(
                    self._action(mode_id=3, q_e4=5000),
                    frame_id=ticket.reward_carla_frame_id,
                ),
                obligation=obligation,
                completed_ticket=ticket,
            )
        message = str(caught.exception)
        self.assertIn("nearest anchor", message)
        self.assertIn("protocol-v2", message)
        nearest = self.contract.find_anchor(
            off_anchor.family, off_anchor.quantizer, 5000
        )
        self.assertIsNotNone(nearest)
        self.assertNotIn(str(nearest.action_id), message.split("action_id")[0])

        # evidence claiming per-frame causality without an ACK is refused
        with self.assertRaises(src.QualityContractError):
            src.QualityEvidenceV1(
                gt_source=src.GroundTruthSource.CARLA_GT_EXACT,
                kind=src.EvidenceKind.PER_FRAME_CAUSAL_ACK,
                granularity=src.EvidenceGranularity.SINGLE_FRAME,
                eligibility=self._eligibility(),
                executed_action_sha256=off_anchor.canonical_sha256(),
                gt_source_detail="x",
                ack_binding=None,
            )
        # aggregate evidence exists but can never earn a per-frame reward
        aggregate = src.QualityEvidenceV1(
            gt_source=src.GroundTruthSource.CARLA_GT_EXACT,
            kind=src.EvidenceKind.AGGREGATE_PROFILE_CAMPAIGN,
            granularity=src.EvidenceGranularity.PROFILE_AGGREGATE,
            eligibility=self._eligibility(),
            executed_action_sha256=off_anchor.canonical_sha256(),
            gt_source_detail="288-cell campaign aggregate",
            ack_binding=None,
        )
        self.assertFalse(aggregate.is_causal_per_frame)
        with self.assertRaises(src.EvidenceGranularityError) as caught:
            aggregate.require_causal_per_frame()
        self.assertIn("action average", str(caught.exception))
        with self.assertRaises(src.QualityContractError):
            src.QualityComponentsV1.from_ack_binding(
                aggregate,
                pred_vehicle_pixels=1,
                pred_person_pixels=1,
                vehicle_eligible_gt_instances=1,
                person_eligible_gt_instances=1,
            )
        # an exact-feedback ticket with no ACK binding is not learning eligible
        anchor_ticket = self._ticket()
        with self.assertRaises(src.QualityContractError):
            src.evaluate_completed_decision(
                anchor_ticket, self._reward_spec(), quality_components=None
            )

    # ------------------------------------------------------------------ 4 -- #

    def test_reject_gt_present_person_miss_being_masked_or_rewarded(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        document = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                person={"tp": 0, "fn": 2, "recall": 0.0,
                        "source_time_world_xy_error_m": None}
            ),
        )
        evidence = self._evidence(
            ticket,
            obligation=obligation,
            binding=self._binding(
                ticket, document=document, obligation=obligation
            ),
        )
        # the missed eligible person stays in the measurement
        components = self._components(
            evidence=evidence, person_eligible_gt_instances=2
        )
        self.assertTrue(components.person_localization.is_defined)
        self.assertIn("person", components.defined_localization_classes)
        self.assertIn("person", components.missed_localization_classes)
        self.assertNotIn("person", components.undefined_localization_classes)
        evaluated = spec.evaluate_quality(components)
        self.assertIn("person", evaluated.localization_weights_used)
        self.assertEqual(evaluated.per_class_localization_utility["person"], 0.0)
        # it is neither renormalized away nor rewarded
        self.assertEqual(evaluated.q_perc, 0.0)
        vehicle_only = dict(evaluated.localization_weights_used)
        self.assertEqual(sorted(vehicle_only), ["person", "vehicle"])

        # a miss cannot be laundered by understating the eligible count
        with self.assertRaises(src.QualityContractError) as caught:
            self._components(evidence=evidence, person_eligible_gt_instances=0)
        self.assertIn("eligible ground-truth count", str(caught.exception))
        with self.assertRaises(src.QualityContractError):
            self._components(evidence=evidence, person_eligible_gt_instances=1)

        # nor by inventing a matched error for an unmatched class
        bad_document = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                person={"tp": 0, "fn": 2, "recall": 0.0,
                        "source_time_world_xy_error_m": 0.0}
            ),
        )
        bad_evidence = self._evidence(
            ticket,
            obligation=obligation,
            binding=self._binding(
                ticket, document=bad_document, obligation=obligation
            ),
        )
        with self.assertRaises(src.QualityContractError) as caught:
            self._components(evidence=bad_evidence)
        self.assertIn("no matched object", str(caught.exception))

        # recall > 0 requires a finite matched error
        missing_error = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                person={"tp": 1, "fn": 1, "source_time_world_xy_error_m": None}
            ),
        )
        with self.assertRaises(src.UndefinedClassSupportError):
            self._components(
                evidence=self._evidence(
                    ticket,
                    obligation=obligation,
                    binding=self._binding(
                        ticket, document=missing_error, obligation=obligation
                    ),
                )
            )

        # GT presence with zero ELIGIBLE objects is excluded, not penalized:
        # the object was out of range / FoV / occluded for this UE.
        ineligible_doc = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                person={"tp": 0, "fn": 0, "recall": None,
                        "source_time_world_xy_error_m": None}
            ),
        )
        ineligible = self._components(
            evidence=self._evidence(
                ticket,
                obligation=obligation,
                binding=self._binding(
                    ticket, document=ineligible_doc, obligation=obligation
                ),
            ),
            person_eligible_gt_instances=0,
        )
        self.assertFalse(ineligible.person_localization.is_defined)
        self.assertIn("person", ineligible.undefined_localization_classes)
        excluded = spec.evaluate_quality(ineligible)
        self.assertNotIn("person", excluded.localization_weights_used)
        self.assertGreater(excluded.q_perc, 0.0)
        # the eligibility rule and its hash travel with the record
        self.assertEqual(
            ineligible.eligibility.eligibility_contract_sha256, HEX_A
        )
        self.assertIs(
            ineligible.eligibility.reward_scope,
            src.RewardScope.PER_UE_PERCEPTION,
        )
        self.assertFalse(
            ineligible.to_canonical_dict()["evidence"]["eligibility"][
                "gt_presence_alone_penalizes"
            ]
        )

    # ------------------------------------------------------------------ 5 -- #

    def test_reject_gt_absent_segmentation_false_positive_exclusion(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        document = self._ack_document(
            ticket.action,
            frame_id=ticket.reward_carla_frame_id,
            quality=self._ack_quality(
                segmentation={"gt_person_pixels": 0, "miou_person_iou": 0.0}
            ),
        )
        evidence = self._evidence(
            ticket,
            obligation=obligation,
            binding=self._binding(
                ticket, document=document, obligation=obligation
            ),
        )
        # a predicted mask against absent GT is NOT excluded
        components = self._components(evidence=evidence, pred_person_pixels=450)
        self.assertTrue(components.person_segmentation.is_defined)
        self.assertIsNone(components.person_segmentation.exclusion_reason)
        evaluated = spec.evaluate_quality(components)
        self.assertIn("person", evaluated.segmentation_weights_used)
        self.assertEqual(evaluated.q_seg, 0.0)
        # and it demonstrably costs quality relative to no false positive
        clean = spec.evaluate_quality(
            self._components(evidence=evidence, pred_person_pixels=0)
        )
        self.assertFalse(clean.components.person_segmentation.is_defined)
        self.assertGreater(clean.q_perc, evaluated.q_perc)

        # an excluded class may not smuggle a non-zero IoU
        with self.assertRaises(src.QualityContractError):
            src._ClassSegmentation(
                name="person", iou=0.4, gt_pixels=0, pred_pixels=0
            )
        # a defined class must carry an IoU
        with self.assertRaises(src.QualityContractError):
            src._ClassSegmentation(
                name="person", iou=None, gt_pixels=10, pred_pixels=0
            )
        # unmasked segmentation is refused outright
        with self.assertRaises(src.QualityContractError) as caught:
            self._eligibility(segmentation_eligibility_masked=False)
        self.assertIn("eligibility-masked", str(caught.exception))

    # ------------------------------------------------------------------ 6 -- #

    def test_reject_forged_quality_or_scalar_reward(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        components = self._components(ticket)
        real = spec.evaluate_quality(components)

        # a directly constructed evaluation is unattested
        forged = src.QualityEvaluationV1(
            components=components,
            q_seg=1.0,
            q_loc=1.0,
            q_perc=1.0,
            per_class_localization_utility={"vehicle": 1.0},
            per_class_xy_utility={"vehicle": 1.0},
            per_class_recall={"vehicle": 1.0},
            per_class_normalized_segmentation={"vehicle": 1.0},
            segmentation_weights_used={"vehicle": 1.0},
            localization_weights_used={"vehicle": 1.0},
            localization_combiner=spec.localization_combiner,
            reward_spec_sha256=spec.canonical_sha256(),
        )
        self.assertFalse(forged.is_attested)
        with self.assertRaises(src.UnattestedRecordError):
            forged.to_canonical_dict()
        with self.assertRaises(src.UnattestedRecordError):
            src.DecisionOutcomeV1(
                terminal_class=ticket.terminal_class,
                eligibility=src.LearningEligibility.ELIGIBLE,
                costs=src.ConstraintCostsV1(0.0, 0.0),
                diagnostics=src.DiagnosticSignalsV1(0.0),
                switch_penalty=src.SwitchPenaltyV1.between(
                    None, ticket.action, spec
                ),
                completed_ticket_sha256=ticket.canonical_sha256(),
                reward_spec_sha256=spec.canonical_sha256(),
                quality=forged,
                scalar_reward=99.0,
            )

        # an attested evaluation cannot be mutated at all: the attestation is
        # bound to its serialized fields and is refused at construction, so a
        # tampered copy cannot even come into existence
        import dataclasses

        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(real, q_perc=1.0)
        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(real, q_loc=0.0)
        # negative or zero component weights are refused outright
        for weights in ({"vehicle": 0.0}, {"vehicle": -1.0}):
            with self.subTest(weights=weights):
                with self.assertRaises(src.QualityContractError):
                    src.QualityEvaluationV1(
                        components=components,
                        q_seg=0.5, q_loc=0.5, q_perc=0.5,
                        per_class_localization_utility={"vehicle": 0.5},
                        per_class_xy_utility={"vehicle": 0.5},
                        per_class_recall={"vehicle": 1.0},
                        per_class_normalized_segmentation={"vehicle": 0.5},
                        segmentation_weights_used={"vehicle": 1.0},
                        localization_weights_used=weights,
                        localization_combiner=spec.localization_combiner,
                        reward_spec_sha256=spec.canonical_sha256(),
                    )

        # a transition carrying a forged outcome fails re-derivation
        transition = self._full_transition()
        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(transition.outcome, scalar_reward=99.0)
        # a transition cannot be mutated either
        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(
                transition, freshness_policy_sha256=HEX_A
            )
        # and a transition built under a different spec than its outcome fails
        with self.assertRaises(src.TransitionIdentityError):
            src.build_replay_transition(
                state=transition.state,
                next_state=transition.next_state,
                completed_ticket=transition.completed_ticket,
                outcome=transition.outcome,
                policy_trace=transition.policy_trace,
                reward_spec=self._reward_spec(w_quality=2.0),
                normalization=self._norm(),
                freshness=self._freshness(),
            )

    # ------------------------------------------------------------------ 7 -- #

    def test_reject_eligible_timeout_without_bound_adjudication(self) -> None:
        spec = self._reward_spec()
        timeout = self._ticket(terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT)

        # without adjudication it stays censored and unrewarded
        censored = src.evaluate_completed_decision(timeout, spec)
        self.assertIsNone(censored.scalar_reward)
        self.assertFalse(censored.learning_eligible)

        # a PENDING verdict changes nothing
        pending = src.evaluate_completed_decision(
            timeout,
            spec,
            adjudication=self._adjudication(timeout, src.Adjudication.PENDING),
        )
        self.assertIsNone(pending.scalar_reward)

        # an outcome asserting eligibility with no adjudication is unattested,
        # and fails re-derivation inside a transition
        import dataclasses

        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(
                censored,
                eligibility=src.LearningEligibility.ELIGIBLE,
                scalar_reward=-1.0,
            )
        # nor can a fresh outcome simply assert eligibility with a reward
        with self.assertRaises(src.UnattestedRecordError):
            src.DecisionOutcomeV1(
                terminal_class=rtc.TerminalClass.FEEDBACK_TIMEOUT,
                eligibility=src.LearningEligibility.ELIGIBLE,
                costs=src.ConstraintCostsV1(1.0, 1.0),
                diagnostics=src.DiagnosticSignalsV1(None),
                switch_penalty=src.SwitchPenaltyV1.between(
                    None, timeout.action, spec
                ),
                completed_ticket_sha256=timeout.canonical_sha256(),
                reward_spec_sha256=spec.canonical_sha256(),
                scalar_reward=-1.0,
                _attestation=("forged", "token"),
            )

        # an adjudication predating closure is refused
        with self.assertRaises(src.AdjudicationError) as caught:
            src.AdjudicationRecordV1.for_ticket(
                timeout,
                verdict=src.Adjudication.AUTHORITATIVE_SERVICE_FAILURE,
                adjudicator_id="x",
                evidence_sha256=HEX_C,
                adjudicated_ns=timeout.closed_ns - 1,
                detail="too early",
            )
        self.assertIn("predates", str(caught.exception))

        # adjudication is meaningless for an already-authoritative terminal
        for terminal in (
            rtc.TerminalClass.REWARD_FINAL_EXACT,
            rtc.TerminalClass.ACTION_PATH_FAILURE,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        ):
            with self.subTest(terminal=terminal):
                other = self._ticket(terminal=terminal)
                with self.assertRaises(src.AdjudicationError):
                    src.evaluate_completed_decision(
                        other,
                        spec,
                        quality_components=(
                            self._components(other)
                            if terminal is rtc.TerminalClass.REWARD_FINAL_EXACT
                            else None
                        ),
                        adjudication=self._adjudication(
                            other, src.Adjudication.FEEDBACK_ONLY_LOSS
                        ),
                    )

    # ------------------------------------------------------------------ 8 -- #

    def test_reject_adjudication_reused_for_another_ticket(self) -> None:
        spec = self._reward_spec()
        first = self._ticket(
            terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT,
            decision_seq=1,
            first_tensor_seq=10,
            first_frame_id=500,
        )
        second = self._ticket(
            terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT,
            decision_seq=2,
            first_tensor_seq=20,
            first_frame_id=600,
        )
        verdict = self._adjudication(
            first, src.Adjudication.AUTHORITATIVE_SERVICE_FAILURE
        )
        verdict.assert_binds(first)
        self.assertEqual(
            verdict.completed_ticket_sha256, first.canonical_sha256()
        )

        with self.assertRaises(src.AdjudicationError) as caught:
            verdict.assert_binds(second)
        self.assertIn("decision", str(caught.exception))
        with self.assertRaises(src.AdjudicationError):
            src.evaluate_completed_decision(
                second, spec, adjudication=verdict
            )

        # a hand-built record with the right seqs but the wrong ticket hash
        import dataclasses

        relabelled = dataclasses.replace(
            verdict,
            decision_seq=second.decision_seq,
            session_uuid=second.session_uuid,
        )
        with self.assertRaises(src.AdjudicationError) as caught:
            relabelled.assert_binds(second)
        self.assertIn("not reusable across tickets", str(caught.exception))
        # a cross-session verdict is refused
        cross = dataclasses.replace(verdict, session_uuid=OTHER_SESSION)
        with self.assertRaises(src.AdjudicationError):
            cross.assert_binds(first)
        # and a verdict issued for a different terminal class
        wrong_terminal = dataclasses.replace(
            verdict, terminal_class=rtc.TerminalClass.ACTION_PATH_FAILURE
        )
        with self.assertRaises(src.AdjudicationError):
            wrong_terminal.assert_binds(first)

    # ------------------------------------------------------------------ 9 -- #

    def test_reject_state_observed_after_ticket_opening(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket(opened_ns=T0)
        components = self._components(ticket)
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=components
        )
        late_state = self._state(
            observed_ns=ticket.opened_ns + 1,
            scene=self._scene(measured_ns=ticket.opened_ns + 1),
            radio=self._radio(
                snr_measured_ns=ticket.opened_ns + 1,
                bsr_measured_ns=ticket.opened_ns + 1,
                mcs_measured_ns=ticket.opened_ns + 1,
            ),
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=late_state,
                next_state=self._next_state(ticket, outcome, spec),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=self._norm(),
                freshness=self._freshness(),
            )
        self.assertIn("precedes", str(caught.exception))

        # a measurement postdating the observation is refused at state level
        with self.assertRaises(src.CausalStateError) as caught:
            self._state(scene=self._scene(measured_ns=T0 + 1))
        self.assertIn("postdates", str(caught.exception))
        for field in ("snr_measured_ns", "bsr_measured_ns", "mcs_measured_ns"):
            with self.subTest(field=field):
                with self.assertRaises(src.CausalStateError):
                    self._state(radio=self._radio(**{field: T0 + 1}))
        # and a scene descriptor from another frame cannot be reused
        with self.assertRaises(src.CausalStateError) as caught:
            self._state(scene=self._scene(carla_frame_id=999))
        self.assertIn("never reused across frames", str(caught.exception))

    # ----------------------------------------------------------------- 10 -- #

    def test_reject_next_state_observed_before_closure(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._components(ticket)
        )
        early = self._next_state(
            ticket, outcome, spec, observed_ns=ticket.closed_ns - 1
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=self._state(),
                next_state=early,
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=self._norm(),
                freshness=self._freshness(),
            )
        self.assertIn("closed at", str(caught.exception))

        # and a successor that does not follow every held tensor
        stale_seq = self._next_state(
            ticket, outcome, spec, tensor_seq=ticket.tensor_seqs[-1]
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=self._state(),
                next_state=stale_seq,
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=self._norm(),
                freshness=self._freshness(),
            )
        self.assertIn("must follow every tensor", str(caught.exception))

    # ----------------------------------------------------------------- 11 -- #

    def test_reject_missing_unrelated_or_future_next_previous(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket()
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._components(ticket)
        )

        def _build(next_state: Optional[src.CausalStateV1]) -> None:
            src.build_replay_transition(
                state=self._state(),
                next_state=next_state,
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=self._norm(),
                freshness=self._freshness(),
            )

        # absent previous
        with self.assertRaises(src.TransitionIdentityError) as caught:
            _build(self._next_state(ticket, outcome, spec, previous=None))
        self.assertIn("is absent", str(caught.exception))

        # unrelated previous: a different decision's completed outcome
        other = self._ticket(
            decision_seq=5, first_tensor_seq=40, first_frame_id=700
        )
        other_outcome = src.evaluate_completed_decision(
            other, spec, quality_components=self._components(other)
        )
        unrelated = src.PreviousOutcomeV1.from_completed(
            other, other_outcome, spec
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            _build(self._next_state(ticket, outcome, spec, previous=unrelated))
        self.assertIn("bound to ticket", str(caught.exception))

        # a cross-session next state.  It must be internally consistent -- a
        # state refuses a previous outcome from another session outright -- so
        # the successor is built wholly inside the other session, which is
        # exactly the cross-session splice the transition must refuse.
        foreign_ticket = self._ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500,
            session_uuid=OTHER_SESSION,
        )
        foreign_outcome = src.evaluate_completed_decision(
            foreign_ticket,
            spec,
            quality_components=self._components(foreign_ticket),
        )
        foreign_next = self._next_state(
            foreign_ticket, foreign_outcome, spec
        )
        self.assertEqual(foreign_next.session_uuid, OTHER_SESSION)
        with self.assertRaises(src.TransitionIdentityError) as caught:
            _build(foreign_next)
        self.assertIn("crosses out of session", str(caught.exception))

        # a state carrying its own or a future decision as "previous"
        self_previous = src.PreviousOutcomeV1.from_completed(
            ticket, outcome, spec
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=self._state(previous=self_previous),
                next_state=self._next_state(ticket, outcome, spec),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=self._norm(),
                freshness=self._freshness(),
            )
        self.assertIn("must precede", str(caught.exception))

        # a forged previous-outcome record cannot be built at all
        import dataclasses

        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(self_previous, quality_normalized=1.0)
        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(self_previous, decision_seq=99)
        # and a hand-built one cannot enter a state
        hand_built = src.PreviousOutcomeV1(
            session_uuid=SESSION,
            decision_seq=0,
            action=ticket.action,
            terminal_class=rtc.TerminalClass.REWARD_FINAL_EXACT,
            eligibility=src.LearningEligibility.ELIGIBLE.value,
            completed_ticket_sha256=ticket.canonical_sha256(),
            outcome_sha256=outcome.canonical_sha256(),
            reward_spec_sha256=spec.canonical_sha256(),
            resolution_ns=ticket.resolution_ns,
            quality_normalized=1.0,
            latency_normalized=0.0,
            quality_gt_source="CARLA_GT_EXACT",
            latency_clock_domain="UE_LOCAL_MONOTONIC",
        )
        self.assertFalse(hand_built.is_attested)
        with self.assertRaises(src.UnattestedRecordError):
            self._state(previous=hand_built)
        # and an outcome/ticket mismatch cannot produce one
        with self.assertRaises(src.CausalStateError):
            src.PreviousOutcomeV1.from_completed(other, outcome, spec)
        with self.assertRaises(src.CausalStateError):
            src.PreviousOutcomeV1.from_completed(
                ticket, outcome, self._reward_spec(w_quality=3.0)
            )

    # ----------------------------------------------------------------- 12 -- #

    def test_reject_caller_forged_zero_ages(self) -> None:
        # there is no age argument anywhere: ages are derived only
        import inspect

        for record in (src.CausalStateV1, src.SceneObservationV1,
                       src.RadioObservationV1):
            with self.subTest(record=record.__name__):
                parameters = set(
                    inspect.signature(record.__init__).parameters
                )
                for forbidden in (
                    "scene_age_ns", "snr_age_ns", "bsr_age_ns", "mcs_age_ns",
                    "age_ns", "ages",
                ):
                    self.assertNotIn(forbidden, parameters)

        # the only way to get a zero age is a measurement at the observation
        fresh_state = self._state(
            scene=self._scene(measured_ns=T0),
            radio=self._radio(
                snr_measured_ns=T0, bsr_measured_ns=T0, mcs_measured_ns=T0
            ),
        )
        self.assertEqual(
            dict(fresh_state.measurement_ages_ns),
            {"scene": 0, "snr": 0, "bsr": 0, "mcs": 0},
        )
        named = src.build_policy_features(
            fresh_state, self._norm(), self._freshness()
        ).as_mapping()
        for key in (
            "freshness_scene_normalized", "freshness_snr_normalized",
            "freshness_bsr_normalized", "freshness_mcs_normalized",
        ):
            self.assertEqual(named[key], 0.0)

        # a genuinely stale state cannot be vectorized at all
        fresh = self._freshness()
        stale = self._state(
            scene=self._scene(measured_ns=T0 - fresh.max_scene_age_ns - 1)
        )
        with self.assertRaises(src.StaleTelemetryError) as caught:
            src.build_policy_features(stale, self._norm(), fresh)
        self.assertIn("registered fallback", str(caught.exception))

        # mismatched radio semantics cannot be scaled by the wrong constants
        for override, expected in (
            ({"snr_metric": src.SnrMetric.UL_PUCCH_SNR_DB}, "metric"),
            ({"mcs_table_id": "oai_ul_table_2"}, "table"),
            (
                {"bsr_scope": src.BsrScope.LOGICAL_CHANNEL_GROUP_LATEST},
                "BSR",
            ),
        ):
            with self.subTest(override=override):
                with self.assertRaises(src.NormalizationSpecError) as caught:
                    src.build_policy_features(
                        self._state(radio=self._radio(**override)),
                        self._norm(),
                        fresh,
                    )
                self.assertIn(expected, str(caught.exception))
        # downlink measurements are not the reward-facing quantity
        for field in ("snr_direction", "mcs_direction"):
            with self.subTest(field=field):
                with self.assertRaises(src.CausalStateError):
                    self._radio(**{field: src.LinkDirection.DOWNLINK})

    # ----------------------------------------------------------------- 13 -- #

    def test_reject_freshness_id_reused_with_changed_bounds(self) -> None:
        original = self._freshness()
        relabelled = self._freshness(max_snr_age_ns=999 * MS)
        # the same free-form id, different bounds -> different hash
        self.assertEqual(original.policy_id, relabelled.policy_id)
        self.assertNotEqual(
            original.canonical_sha256(), relabelled.canonical_sha256()
        )

        state = self._state()
        norm = self._norm()
        first = src.build_policy_features(state, norm, original)
        second = src.build_policy_features(state, norm, relabelled)
        # the id alone would have looked identical; the hash does not
        self.assertEqual(first.freshness_policy_id, second.freshness_policy_id)
        self.assertNotEqual(
            first.freshness_policy_sha256, second.freshness_policy_sha256
        )
        # and the normalized age features actually differ
        self.assertNotEqual(
            first.as_mapping()["freshness_snr_normalized"],
            second.as_mapping()["freshness_snr_normalized"],
        )

        # a transition binds the complete hash, so the two are distinguishable
        spec = self._reward_spec()
        ticket = self._ticket()
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._components(ticket)
        )
        transitions = [
            src.build_replay_transition(
                state=self._state(),
                next_state=self._next_state(ticket, outcome, spec),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(ticket.action),
                reward_spec=spec,
                normalization=norm,
                freshness=policy,
            )
            for policy in (original, relabelled)
        ]
        self.assertNotEqual(
            transitions[0].freshness_policy_sha256,
            transitions[1].freshness_policy_sha256,
        )
        self.assertNotEqual(
            transitions[0].canonical_sha256(), transitions[1].canonical_sha256()
        )
        # the same applies to the normalization spec
        self.assertNotEqual(
            self._norm(camera_si_clip_max=120.0).canonical_sha256(),
            norm.canonical_sha256(),
        )

    # ----------------------------------------------------------------- 14 -- #

    def test_reject_sampled_and_executed_q_mismatch(self) -> None:
        action = self._action(mode_id=3, q_e4=9800)

        # the honest sample quantizes to the executed q_e4
        self._trace(action, sampled_q=0.98)
        self._trace(action, sampled_q=0.97996)

        # a sample that would quantize elsewhere is refused
        for bad_q in (0.5, 0.97, 0.0, 0.9794):
            with self.subTest(sampled_q=bad_q):
                with self.assertRaises(src.StateRewardContractError) as caught:
                    self._trace(action, sampled_q=bad_q)
                self.assertIn("quantizes to", str(caught.exception))

        # a sampled mode different from the executed mode is refused
        with self.assertRaises(src.StateRewardContractError) as caught:
            self._trace(action, sampled_mode_id=7)
        self.assertIn("sampled joint mode", str(caught.exception))

        # log-probabilities must be non-positive
        for field in ("log_prob_discrete", "log_prob_continuous"):
            with self.subTest(field=field):
                with self.assertRaises(src.StateRewardContractError):
                    self._trace(action, **{field: 0.5})

        # and the trace must describe the action the hold actually executed
        spec = self._reward_spec()
        ticket = self._ticket(action=action)
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._components(ticket)
        )
        other_action = self._action(mode_id=5, q_e4=5000)
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=self._state(),
                next_state=self._next_state(ticket, outcome, spec),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=self._trace(other_action),
                reward_spec=spec,
                normalization=self._norm(),
                freshness=self._freshness(),
            )
        self.assertIn("different executed action", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

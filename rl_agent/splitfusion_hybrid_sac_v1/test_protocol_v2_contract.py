"""Offline adversarial tests for :mod:`protocol_v2_contract`.

The tests use tiny byte strings as source artifacts.  They launch no live
runtime and exercise no CARLA, CUDA, radio, container or filesystem path.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from typing import Dict, Tuple

from . import action_contract as ac
from . import protocol_v2_contract as p2
from . import reward_ticket_controller as rtc
from . import transaction_identity as ti


SESSION = "3f263fce-cc44-476e-93b5-19d09d439471"
OTHER_SESSION = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
LINEAGE = "6d5ef476-4ea5-4bf0-9db6-65c15ac06936"
TRACE_SHA = "a" * 64
REWARD_SPEC_SHA = "b" * 64
EVIDENCE_SHA = "c" * 64
RECONCILER_SHA = "d" * 64
T0 = 1_000_000_000


def _independent_bytes(payload) -> bytes:
    """Independent canonicalizer (no call into the implementation helper)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class ProtocolV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()

    def _action(
        self, mode_id: int = 8, q_e4: int = 3333
    ) -> ti.ExecutedActionIdentity:
        executable = self.contract.resolve(mode_id, q_e4 / ac.Q_E4_SCALE)
        self.assertEqual(executable.q_e4, q_e4)
        return ti.ExecutedActionIdentity.from_executable_action(
            executable, self.contract
        )

    def _request(
        self,
        *,
        session_uuid: str = SESSION,
        decision_seq: int = 7,
        tensor_seq: int = 19,
        frame_id: int = 401,
        action: ti.ExecutedActionIdentity | None = None,
        reward_requested: bool = True,
    ) -> p2.DynamicFeatureRequestV2:
        chosen = self._action() if action is None else action
        envelope = ti.TensorTransmissionEnvelope(
            transaction=ti.TensorTransactionId(
                session_uuid=session_uuid,
                decision_seq=decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=frame_id,
            ),
            reward_requested=reward_requested,
            action=chosen,
        )
        return p2.DynamicFeatureRequestV2(
            run_id="run-v2",
            cell_id="cell-v2",
            stream_id="ue-a",
            controller_lineage_uuid=LINEAGE,
            capture_timestamp_ns=123_456_789,
            policy_decision_trace_sha256=TRACE_SHA,
            envelope=envelope,
        )

    def _quality(self, q_perc: float = 0.73) -> p2.ExactPerceptionQualityV2:
        spec = self._reward_spec()
        return p2.ExactPerceptionQualityV2(
            seg_vehicle_iou=0.81,
            seg_person_iou=0.55,
            vehicle_recall=0.90,
            person_recall=0.75,
            vehicle_xy_error_m=0.42,
            person_xy_error_m=0.61,
            q_seg=0.67,
            q_loc=0.79,
            q_perc=q_perc,
            reward_spec=spec,
        )

    def _reward_spec(self):
        from . import state_reward_transition_contract as sr

        return sr.RewardSpecV1(
            spec_id="protocol-v2-test",
            spec_version=1,
            w_loc_person=0.6,
            w_loc_vehicle=0.4,
            tau_person_m=2.0,
            tau_vehicle_m=3.0,
            localization_combiner=sr.LocalizationCombiner.WEIGHTED_GEOMETRIC_MEAN,
            w_seg_person=0.6,
            w_seg_vehicle=0.4,
            seg_reference_person_iou=0.6,
            seg_reference_vehicle_iou=0.8,
            segmentation_modulation_beta=0.3,
            w_quality=1.0,
            w_latency=0.5,
            lambda_mode=0.01,
            lambda_q=0.01,
            r_registered_failure=-1.0,
            gamma_per_tensor=0.99,
            provenance={"test": "protocol-v2"},
        )

    def _raw_artifacts(
        self,
        request: p2.DynamicFeatureRequestV2,
        *,
        person_intersection: int = 30,
    ) -> Tuple[
        Dict[p2.CarlaSourceArtifactRole, bytes],
        p2.ExactPerceptionQualityV2,
    ]:
        artifacts = {
            role: ("raw:" + role.value).encode("ascii")
            for role in p2.REQUIRED_CARLA_SOURCE_ARTIFACT_ROLES
        }
        artifacts[
            p2.CarlaSourceArtifactRole.FEATURE_REQUEST_CANONICAL_JSON
        ] = request.canonical_bytes()
        artifacts[p2.CarlaSourceArtifactRole.PRODUCER_SOURCE_BYTES] = (
            p2._REVIEWED_PRODUCER_SOURCE_BYTES_V2
        )
        frame_id = request.transaction.carla_frame_id
        artifacts[p2.CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON] = (
            ti.canonical_json_bytes(
                {
                    "actors": [
                        {"actor_id": 10},
                        {"actor_id": 11},
                        {"actor_id": 20},
                    ],
                    "carla_frame_id": frame_id,
                    "record": "carla_actor_snapshot_v2",
                }
            )
        )
        artifacts[
            p2.CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON
        ] = ti.canonical_json_bytes(
            {
                "carla_frame_id": frame_id,
                "eligible_person_actor_ids": [20],
                "eligible_vehicle_actor_ids": [10, 11],
                "record": "actor_projection_eligibility_v2",
            }
        )
        artifacts[p2.CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON] = (
            ti.canonical_json_bytes(
                {"record": "camera_calibration_v2", "sha256": "1" * 64}
            )
        )
        artifacts[p2.CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON] = (
            ti.canonical_json_bytes(
                {"record": "eligibility_contract_v2", "sha256": "2" * 64}
            )
        )
        artifacts[p2.CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON] = (
            ti.canonical_json_bytes(
                {
                    "carla_frame_id": frame_id,
                    "objects": [
                        {"class_name": "vehicle", "prediction_id": 1},
                        {"class_name": "person", "prediction_id": 2},
                    ],
                    "record": "prediction_object_records_v2",
                }
            )
        )
        segmentation = {
            "camera_calibration_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON]
            ).hexdigest(),
            "carla_frame_id": frame_id,
            "carla_semantic_label_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.CARLA_SEMANTIC_LABEL]
            ).hexdigest(),
            "eligibility_contract_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON]
            ).hexdigest(),
            "person": {
                "gt_pixels": 50,
                "intersection_pixels": person_intersection,
                "pred_pixels": 40,
                "union_pixels": 90 - person_intersection,
            },
            "prediction_segmentation_label_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.PREDICTION_SEGMENTATION_LABEL]
            ).hexdigest(),
            "record": "segmentation_sufficient_counts_v2",
            "segmentation_eligibility_mask_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.SEGMENTATION_ELIGIBILITY_MASK]
            ).hexdigest(),
            "vehicle": {
                "gt_pixels": 100,
                "intersection_pixels": 80,
                "pred_pixels": 90,
                "union_pixels": 110,
            },
        }
        localization = {
            "actor_projection_eligibility_sha256": hashlib.sha256(
                artifacts[
                    p2.CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON
                ]
            ).hexdigest(),
            "actor_snapshot_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON]
            ).hexdigest(),
            "camera_calibration_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON]
            ).hexdigest(),
            "carla_frame_id": frame_id,
            "eligibility_contract_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON]
            ).hexdigest(),
            "person": {
                "eligible_actor_ids": [20],
                "eligible_gt_instances": 1,
                "fn": 0,
                "matched_xy_errors_m": [0.6],
                "tp": 1,
            },
            "prediction_object_records_sha256": hashlib.sha256(
                artifacts[p2.CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON]
            ).hexdigest(),
            "record": "localization_match_ledger_v2",
            "vehicle": {
                "eligible_actor_ids": [10, 11],
                "eligible_gt_instances": 2,
                "fn": 0,
                "matched_xy_errors_m": [0.3, 0.5],
                "tp": 2,
            },
        }
        artifacts[
            p2.CarlaSourceArtifactRole.SEGMENTATION_SUFFICIENT_COUNTS_JSON
        ] = ti.canonical_json_bytes(segmentation)
        artifacts[
            p2.CarlaSourceArtifactRole.LOCALIZATION_MATCH_LEDGER_JSON
        ] = ti.canonical_json_bytes(localization)
        quality = p2._derive_exact_quality(
            self._reward_spec(), segmentation, localization
        )
        artifacts[
            p2.CarlaSourceArtifactRole.QUALITY_EVALUATION_CANONICAL_JSON
        ] = quality.canonical_bytes()
        return artifacts, quality

    def _manifest(
        self,
        request: p2.DynamicFeatureRequestV2,
        *,
        person_intersection: int = 30,
    ) -> p2.CarlaSourceManifestV2:
        artifacts, _ = self._raw_artifacts(
            request, person_intersection=person_intersection
        )
        return p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
            producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
            request=request,
            reward_spec=self._reward_spec(),
            raw_artifacts=artifacts,
        )

    def _ack(
        self,
        request: p2.DynamicFeatureRequestV2,
        *,
        person_intersection: int = 30,
    ) -> p2.QualityAckV2:
        return p2.QualityAckV2(
            request=request,
            feedback_identity=request.expected_feedback_identity(),
            terminal_status=rtc.FeedbackTerminalStatus.REWARD_FINAL,
            edge_tail_completed_ns=5_000,
            edge_ack_emitted_ns=5_500,
            source_manifest=self._manifest(
                request, person_intersection=person_intersection
            ),
        )

    def _timed_out_ticket(
        self,
        *,
        session_uuid: str = SESSION,
        decision_seq: int = 7,
        first_tensor_seq: int = 19,
        first_frame_id: int = 401,
        action: ti.ExecutedActionIdentity | None = None,
    ) -> Tuple[rtc.CompletedTicket, rtc.AdmissionResult]:
        chosen = self._action() if action is None else action
        controller = rtc.RewardTicketController(
            session_uuid,
            controller_lineage_uuid=LINEAGE,
        )
        opened = controller.open_decision(
            decision_seq=decision_seq,
            tensor_seq=first_tensor_seq,
            carla_frame_id=first_frame_id,
            action=chosen,
            now_ns=T0,
            policy_decision_trace_sha256=TRACE_SHA,
        )
        controller.reuse_held_action(
            tensor_seq=first_tensor_seq + 1,
            carla_frame_id=first_frame_id + 1,
            now_ns=T0 + 50_000_000,
        )
        status = controller.observe(T0 + rtc.B_REWARD_DEADLINE_NS + 1)
        self.assertIsNotNone(status.completed_ticket)
        ticket = status.completed_ticket
        assert ticket is not None
        self.assertIs(ticket.terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT)
        self.assertTrue(ticket.lineage_is_attested)
        return ticket, opened

    # -- schema and dynamic action --------------------------------------- #

    def test_schema_descriptors_are_immutable_and_hash_bound(self) -> None:
        pairs = (
            (
                p2.DYNAMIC_FEATURE_REQUEST_SCHEMA_DESCRIPTOR,
                p2.DYNAMIC_FEATURE_REQUEST_SCHEMA_SHA256,
            ),
            (
                p2.CARLA_SOURCE_MANIFEST_SCHEMA_DESCRIPTOR,
                p2.CARLA_SOURCE_MANIFEST_SCHEMA_SHA256,
            ),
            (
                p2.QUALITY_ACK_V2_SCHEMA_DESCRIPTOR,
                p2.QUALITY_ACK_V2_SCHEMA_SHA256,
            ),
            (
                p2.TIMEOUT_RECONCILIATION_SCHEMA_DESCRIPTOR,
                p2.TIMEOUT_RECONCILIATION_SCHEMA_SHA256,
            ),
        )
        for descriptor, expected in pairs:
            with self.assertRaises(TypeError):
                descriptor["mutate"] = True
            # Thaw through canonical JSON, then hash with stdlib directly.
            plain = json.loads(ti.canonical_json_bytes(descriptor))
            self.assertEqual(
                hashlib.sha256(_independent_bytes(plain)).hexdigest(), expected
            )

    def test_off_anchor_request_round_trip_keeps_exact_q_and_null_anchor(self) -> None:
        request = self._request()
        self.assertFalse(request.action.is_registered_anchor)
        self.assertEqual(request.action.q_e4, 3333)
        payload = request.to_canonical_dict()
        action = payload["envelope"]["executed_action"]
        self.assertEqual(action["q_e4"], 3333)
        self.assertIsNone(action["action_id"])
        self.assertIsNone(action["profile_id"])
        self.assertEqual(request.canonical_bytes(), _independent_bytes(payload))
        self.assertEqual(
            request.canonical_sha256(),
            hashlib.sha256(request.canonical_bytes()).hexdigest(),
        )

    def test_anchor_substitution_and_non_reward_feedback_fail_closed(self) -> None:
        request = self._request()
        with self.assertRaises(ti.UnreconciledActionIdentityError):
            replace(request.action, action_id=0, profile_id="forged-anchor")
        no_reward = self._request(reward_requested=False)
        with self.assertRaises(p2.ProtocolV2ContractError):
            no_reward.expected_feedback_identity()
        with self.assertRaises(p2.ProtocolV2ContractError):
            _ = no_reward.quality_obligation_sha256
        with self.assertRaises(p2.QualityAckV2Error):
            p2.QualityAckV2(
                request=no_reward,
                feedback_identity=request.expected_feedback_identity(),
                terminal_status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
                edge_tail_completed_ns=1,
                edge_ack_emitted_ns=2,
                failure_evidence_sha256=EVIDENCE_SHA,
            )
    # -- source authentication ------------------------------------------ #

    def test_source_manifest_requires_every_artifact_and_exact_typed_bytes(self) -> None:
        request = self._request()
        missing, _ = self._raw_artifacts(request)
        del missing[p2.CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON]
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=missing,
            )

        wrong_request, _ = self._raw_artifacts(request)
        wrong_request[
            p2.CarlaSourceArtifactRole.FEATURE_REQUEST_CANONICAL_JSON
        ] = b"{}"
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=wrong_request,
            )

        wrong_quality, _ = self._raw_artifacts(request)
        wrong_quality[
            p2.CarlaSourceArtifactRole.QUALITY_EVALUATION_CANONICAL_JSON
        ] = b"{}"
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=wrong_quality,
            )

        valid, _ = self._raw_artifacts(request)
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=object(),
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=valid,
            )
        arbitrary, _ = self._raw_artifacts(request)
        arbitrary[p2.CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON] = (
            b"caller-selected-arbitrary-bytes"
        )
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=arbitrary,
            )

    def test_source_summaries_bind_every_raw_input_and_exact_frame(self) -> None:
        request = self._request()
        mutations = {
            p2.CarlaSourceArtifactRole.CARLA_SEMANTIC_LABEL: b"changed-gt-label",
            p2.CarlaSourceArtifactRole.PREDICTION_SEGMENTATION_LABEL: (
                b"changed-prediction-label"
            ),
            p2.CarlaSourceArtifactRole.SEGMENTATION_ELIGIBILITY_MASK: (
                b"changed-eligibility-mask"
            ),
            p2.CarlaSourceArtifactRole.CAMERA_CALIBRATION_JSON: (
                ti.canonical_json_bytes(
                    {"record": "camera_calibration_v2", "sha256": "9" * 64}
                )
            ),
            p2.CarlaSourceArtifactRole.ELIGIBILITY_CONTRACT_JSON: (
                ti.canonical_json_bytes(
                    {"record": "eligibility_contract_v2", "sha256": "8" * 64}
                )
            ),
            p2.CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON: (
                ti.canonical_json_bytes(
                    {
                        "actors": [
                            {"actor_id": 10},
                            {"actor_id": 11},
                            {"actor_id": 20},
                            {"actor_id": 21},
                        ],
                        "carla_frame_id": request.transaction.carla_frame_id,
                        "record": "carla_actor_snapshot_v2",
                    }
                )
            ),
            p2.CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON: (
                ti.canonical_json_bytes(
                    {
                        "carla_frame_id": request.transaction.carla_frame_id,
                        "eligible_person_actor_ids": [20],
                        "eligible_vehicle_actor_ids": [11, 10],
                        "record": "actor_projection_eligibility_v2",
                    }
                )
            ),
            p2.CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON: (
                ti.canonical_json_bytes(
                    {
                        "carla_frame_id": request.transaction.carla_frame_id,
                        "objects": [],
                        "record": "prediction_object_records_v2",
                    }
                )
            ),
        }
        for role, replacement_bytes in mutations.items():
            with self.subTest(role=role.value):
                artifacts, _ = self._raw_artifacts(request)
                artifacts[role] = replacement_bytes
                with self.assertRaises(p2.SourceAuthenticationError):
                    p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                        producer_capability=(
                            p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2
                        ),
                        request=request,
                        reward_spec=self._reward_spec(),
                        raw_artifacts=artifacts,
                    )

        missing_prediction, _ = self._raw_artifacts(request)
        del missing_prediction[
            p2.CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON
        ]
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=missing_prediction,
            )

        other_request = self._request(
            decision_seq=8, tensor_seq=20, frame_id=402
        )
        other_artifacts, _ = self._raw_artifacts(other_request)
        cross_frame, _ = self._raw_artifacts(request)
        for role in (
            p2.CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON,
            p2.CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON,
            p2.CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON,
            p2.CarlaSourceArtifactRole.SEGMENTATION_SUFFICIENT_COUNTS_JSON,
            p2.CarlaSourceArtifactRole.LOCALIZATION_MATCH_LEDGER_JSON,
            p2.CarlaSourceArtifactRole.QUALITY_EVALUATION_CANONICAL_JSON,
        ):
            cross_frame[role] = other_artifacts[role]
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=cross_frame,
            )

    def test_frame_bearing_source_documents_require_exact_integer_frame(self) -> None:
        request = self._request()
        mutations = (
            (
                p2.CarlaSourceArtifactRole.CARLA_ACTOR_SNAPSHOT_JSON,
                "actor_snapshot_sha256",
            ),
            (
                p2.CarlaSourceArtifactRole.ACTOR_PROJECTION_ELIGIBILITY_JSON,
                "actor_projection_eligibility_sha256",
            ),
            (
                p2.CarlaSourceArtifactRole.PREDICTION_OBJECT_RECORDS_JSON,
                "prediction_object_records_sha256",
            ),
        )
        for role, localization_digest_field in mutations:
            with self.subTest(role=role.value):
                artifacts, _ = self._raw_artifacts(request)
                document = json.loads(artifacts[role])
                document["carla_frame_id"] = float(
                    request.transaction.carla_frame_id
                )
                artifacts[role] = ti.canonical_json_bytes(document)

                ledger_role = (
                    p2.CarlaSourceArtifactRole.LOCALIZATION_MATCH_LEDGER_JSON
                )
                ledger = json.loads(artifacts[ledger_role])
                ledger[localization_digest_field] = hashlib.sha256(
                    artifacts[role]
                ).hexdigest()
                artifacts[ledger_role] = ti.canonical_json_bytes(ledger)

                with self.assertRaises(p2.SourceAuthenticationError):
                    p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                        producer_capability=(
                            p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2
                        ),
                        request=request,
                        reward_spec=self._reward_spec(),
                        raw_artifacts=artifacts,
                    )

    def test_producer_status_is_factory_only_and_manifest_tamper_is_detected(self) -> None:
        request = self._request()
        self.assertFalse(hasattr(p2.CarlaSourceManifestV2, "authenticate"))
        manifest = self._manifest(request)
        self.assertTrue(manifest.is_authenticated)

    def test_quality_is_recomputed_and_numeric_canonicalization_is_unique(self) -> None:
        request = self._request()
        artifacts, derived = self._raw_artifacts(request)
        forged = replace(derived, q_perc=derived.q_perc - 0.01)
        artifacts[
            p2.CarlaSourceArtifactRole.QUALITY_EVALUATION_CANONICAL_JSON
        ] = forged.canonical_bytes()
        with self.assertRaises(p2.SourceAuthenticationError):
            p2.CarlaSourceManifestV2._authenticate_from_reviewed_producer(
                producer_capability=p2._REVIEWED_CARLA_PRODUCER_CAPABILITY_V2,
                request=request,
                reward_spec=self._reward_spec(),
                raw_artifacts=artifacts,
            )

        spec = self._reward_spec()
        integer_zero = p2.ExactPerceptionQualityV2(
            seg_vehicle_iou=0,
            seg_person_iou=0,
            vehicle_recall=0,
            person_recall=0,
            vehicle_xy_error_m=0,
            person_xy_error_m=0,
            q_seg=0,
            q_loc=0,
            q_perc=0,
            reward_spec=spec,
        )
        float_zero = p2.ExactPerceptionQualityV2(
            seg_vehicle_iou=0.0,
            seg_person_iou=0.0,
            vehicle_recall=0.0,
            person_recall=0.0,
            vehicle_xy_error_m=0.0,
            person_xy_error_m=0.0,
            q_seg=0.0,
            q_loc=0.0,
            q_perc=0.0,
            reward_spec=spec,
        )
        self.assertEqual(integer_zero.canonical_bytes(), float_zero.canonical_bytes())
        self.assertEqual(integer_zero.canonical_sha256(), float_zero.canonical_sha256())
        manifest = self._manifest(request)
        self.assertTrue(manifest.privileged)
        self.assertFalse(manifest.deployable)
        self.assertFalse(manifest.learning_ready)
        self.assertIs(
            manifest.producer_status,
            p2.CarlaSourceAuthenticationStatus.
            REVIEWED_CAPABILITY_VERIFIED_NOT_LIVE_INTEGRATED_V2,
        )
        with self.assertRaises(TypeError):
            p2.CarlaSourceManifestV2(
                request=request,
                quality=manifest.quality,
                artifacts=manifest.artifacts,
                producer_status=(
                    p2.CarlaSourceAuthenticationStatus.
                    REVIEWED_CAPABILITY_VERIFIED_NOT_LIVE_INTEGRATED_V2
                ),
            )
        unattested = p2.CarlaSourceManifestV2(
            request=request,
            quality=manifest.quality,
            artifacts=manifest.artifacts,
        )
        with self.assertRaises(p2.SourceAuthenticationError):
            unattested.canonical_sha256()
        first = manifest.artifacts[0]
        corrupted = replace(first, sha256="f" * 64)
        with self.assertRaises(p2.SourceAuthenticationError):
            replace(manifest, artifacts=(corrupted,) + manifest.artifacts[1:])

        # Frozen records still defend against deliberate object.__setattr__
        # tampering at every serialization boundary.
        original_sha = first.sha256
        object.__setattr__(first, "sha256", "e" * 64)
        try:
            with self.assertRaises(p2.SourceAuthenticationError):
                manifest.canonical_sha256()
        finally:
            object.__setattr__(first, "sha256", original_sha)
        self.assertTrue(manifest.is_authenticated)

    # -- quality ACK ----------------------------------------------------- #

    def test_exact_ack_round_trip_is_explicitly_privileged_nondeployable(self) -> None:
        request = self._request()
        ack = self._ack(request)
        payload = ack.to_canonical_dict()
        self.assertTrue(payload["privileged"])
        self.assertFalse(payload["deployable"])
        self.assertEqual(
            payload["source_authentication_status"],
            "REVIEWED_CAPABILITY_VERIFIED_NOT_LIVE_INTEGRATED_V2",
        )
        self.assertFalse(payload["learning_ready"])
        self.assertEqual(
            payload["quality_obligation_sha256"],
            request.canonical_sha256(),
        )
        self.assertEqual(payload["feedback_identity"]["executed_action"]["q_e4"], 3333)
        self.assertEqual(ack.canonical_bytes(), _independent_bytes(payload))

    def test_ack_rejects_identity_source_and_quality_substitution(self) -> None:
        request = self._request()
        other = self._request(decision_seq=8, tensor_seq=20, frame_id=402)
        with self.assertRaises(p2.QualityAckV2Error):
            p2.QualityAckV2(
                request=request,
                feedback_identity=other.expected_feedback_identity(),
                terminal_status=rtc.FeedbackTerminalStatus.REWARD_FINAL,
                edge_tail_completed_ns=1,
                edge_ack_emitted_ns=2,
                source_manifest=self._manifest(request),
            )

        with self.assertRaises(p2.QualityAckV2Error):
            p2.QualityAckV2(
                request=request,
                feedback_identity=request.expected_feedback_identity(),
                terminal_status=rtc.FeedbackTerminalStatus.REWARD_FINAL,
                edge_tail_completed_ns=1,
                edge_ack_emitted_ns=2,
                source_manifest=self._manifest(other),
            )

    def test_action_path_failure_is_fail_closed_until_typed_evidence_exists(self) -> None:
        request = self._request()
        with self.assertRaises(p2.QualityAckV2Error):
            p2.QualityAckV2(
                request=request,
                feedback_identity=request.expected_feedback_identity(),
                terminal_status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
                edge_tail_completed_ns=1,
                edge_ack_emitted_ns=2,
            )
        with self.assertRaises(p2.QualityAckV2Error):
            p2.QualityAckV2(
                request=request,
                feedback_identity=request.expected_feedback_identity(),
                terminal_status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
                edge_tail_completed_ns=1,
                edge_ack_emitted_ns=2,
                failure_evidence_sha256=EVIDENCE_SHA,
            )
        success = self._ack(request)
        with self.assertRaises(p2.QualityAckV2Error):
            replace(
                success,
                terminal_status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
                source_manifest=None,
                failure_evidence_sha256=EVIDENCE_SHA,
            )

    def test_controller_accepts_exact_raw_ack_digest_and_rejects_quality_conflict(self) -> None:
        action = self._action()
        controller = rtc.RewardTicketController(
            SESSION, controller_lineage_uuid=LINEAGE
        )
        opened = controller.open_decision(
            decision_seq=7,
            tensor_seq=19,
            carla_frame_id=401,
            action=action,
            now_ns=T0,
            policy_decision_trace_sha256=TRACE_SHA,
        )
        controller.reuse_held_action(
            tensor_seq=20,
            carla_frame_id=402,
            now_ns=T0 + 50_000_000,
        )
        request = p2.DynamicFeatureRequestV2(
            run_id="run-v2",
            cell_id="cell-v2",
            stream_id="ue-a",
            controller_lineage_uuid=LINEAGE,
            capture_timestamp_ns=123_456_789,
            policy_decision_trace_sha256=TRACE_SHA,
            envelope=opened.envelope,
        )
        ack_a = self._ack(request, person_intersection=30)
        ack_b = self._ack(request, person_intersection=29)
        self.assertNotEqual(ack_a.canonical_sha256(), ack_b.canonical_sha256())
        carrier_a = p2.QualityAckControllerMessageV2.from_raw_ack(ack_a)
        outcome = controller.submit_feedback(carrier_a, T0 + 80_000_000)
        self.assertIs(
            outcome.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        ticket = outcome.completed_ticket
        assert ticket is not None
        self.assertEqual(ticket.accepted_feedback_sha256, ack_a.canonical_sha256())
        self.assertEqual(outcome.message_sha256, ack_a.canonical_sha256())

        carrier_b = p2.QualityAckControllerMessageV2.from_raw_ack(ack_b)
        conflicting = controller.submit_feedback(carrier_b, T0 + 81_000_000)
        self.assertIs(
            conflicting.disposition,
            rtc.FeedbackDisposition.REJECTED_CONFLICTING_DUPLICATE,
        )

    def test_controller_carrier_revalidates_adversarial_field_mutation(self) -> None:
        action = self._action()

        def controller_and_request(decision_seq, tensor_seq, frame_id):
            controller = rtc.RewardTicketController(
                SESSION, controller_lineage_uuid=LINEAGE
            )
            opened = controller.open_decision(
                decision_seq=decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=frame_id,
                action=action,
                now_ns=T0,
                policy_decision_trace_sha256=TRACE_SHA,
            )
            controller.reuse_held_action(
                tensor_seq=tensor_seq + 1,
                carla_frame_id=frame_id + 1,
                now_ns=T0 + 50_000_000,
            )
            request = p2.DynamicFeatureRequestV2(
                run_id="run-v2",
                cell_id="cell-v2",
                stream_id="ue-a",
                controller_lineage_uuid=LINEAGE,
                capture_timestamp_ns=123_456_789,
                policy_decision_trace_sha256=TRACE_SHA,
                envelope=opened.envelope,
            )
            return controller, request

        status_controller, status_request = controller_and_request(7, 19, 401)
        status_ack = self._ack(status_request)
        status_carrier = p2.QualityAckControllerMessageV2.from_raw_ack(
            status_ack
        )
        object.__setattr__(
            status_carrier,
            "terminal_status",
            rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
        )
        for operation in (
            status_carrier.to_canonical_dict,
            status_carrier.canonical_bytes,
            status_carrier.canonical_sha256,
            lambda: status_carrier.terminal_class,
        ):
            with self.assertRaises(p2.QualityAckV2Error):
                operation()
        with self.assertRaises(p2.QualityAckV2Error):
            status_controller.submit_feedback(
                status_carrier, T0 + 80_000_000
            )

        identity_controller, identity_request = controller_and_request(
            8, 30, 500
        )
        identity_ack = self._ack(status_request)
        identity_carrier = p2.QualityAckControllerMessageV2.from_raw_ack(
            identity_ack
        )
        object.__setattr__(
            identity_carrier,
            "identity",
            identity_request.expected_feedback_identity(),
        )
        for operation in (
            identity_carrier.to_canonical_dict,
            identity_carrier.canonical_bytes,
            identity_carrier.canonical_sha256,
            lambda: identity_carrier.session_uuid,
            lambda: identity_carrier.decision_seq,
            lambda: identity_carrier.reward_tensor_seq,
            lambda: identity_carrier.carla_frame_id,
            lambda: identity_carrier.action,
        ):
            with self.assertRaises(p2.QualityAckV2Error):
                operation()
        with self.assertRaises(p2.QualityAckV2Error):
            identity_controller.submit_feedback(
                identity_carrier, T0 + 80_000_000
            )

        # Both controllers remain unresolved and accept their untampered ACKs.
        status_outcome = status_controller.submit_feedback(
            p2.QualityAckControllerMessageV2.from_raw_ack(status_ack),
            T0 + 81_000_000,
        )
        identity_outcome = identity_controller.submit_feedback(
            p2.QualityAckControllerMessageV2.from_raw_ack(
                self._ack(identity_request)
            ),
            T0 + 81_000_000,
        )
        self.assertIs(
            status_outcome.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT
        )
        self.assertIs(
            identity_outcome.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT
        )

    # -- timeout reconciliation ----------------------------------------- #

    def _base_reconciliation(self, verdict, **kwargs):
        ticket, _ = self._timed_out_ticket()
        return p2.TimeoutReconciliationV1(
            completed_ticket=ticket,
            verdict=verdict,
            reconciler_source_sha256=RECONCILER_SHA,
            reconciled_at_wall_ns=9_000,
            **kwargs,
        )

    def test_conservative_adjudication_table_and_required_evidence(self) -> None:
        ticket, opened = self._timed_out_ticket()
        request = p2.DynamicFeatureRequestV2(
            run_id="run-v2",
            cell_id="cell-v2",
            stream_id="ue-a",
            controller_lineage_uuid=LINEAGE,
            capture_timestamp_ns=123_456_789,
            policy_decision_trace_sha256=TRACE_SHA,
            envelope=opened.envelope,
        )
        service_document = ti.canonical_json_bytes(
            {
                "completed_ticket_sha256": ticket.canonical_sha256(),
                "detected_at_ns": T0 + 120_000_000,
                "failure_code": "FEATURE_REASSEMBLY_EXPIRED",
                "record": "service_terminal_evidence_v2",
                "request_sha256": request.canonical_sha256(),
                "stage": p2.ServiceFailureStage.FEATURE_REASSEMBLY.value,
            }
        )
        service_evidence = (
            p2.ServiceTerminalEvidenceV2._verify_from_reviewed_terminal(
                verifier_capability=(
                    p2._REVIEWED_SERVICE_TERMINAL_CAPABILITY_V2
                ),
                completed_ticket=ticket,
                request=request,
                stage=p2.ServiceFailureStage.FEATURE_REASSEMBLY,
                failure_code="FEATURE_REASSEMBLY_EXPIRED",
                detected_at_ns=T0 + 120_000_000,
                terminal_document=service_document,
            )
        )
        service = p2.TimeoutReconciliationV1(
            completed_ticket=ticket,
            verdict=p2.TimeoutVerdict.ACTION_PATH_SERVICE_FAILURE_NEGATIVE,
            reconciler_source_sha256=RECONCILER_SHA,
            reconciled_at_wall_ns=9_000,
            service_terminal_evidence=service_evidence,
        )
        self.assertTrue(service.penalizes_policy)
        self.assertIs(
            service.learning_disposition,
            p2.ReconciliationLearningDisposition.
            INCLUDED_REGISTERED_NEGATIVE_SERVICE_REWARD,
        )

        feedback_loss = self._base_reconciliation(
            p2.TimeoutVerdict.FEEDBACK_ONLY_LOSS_CENSORED,
            edge_quality_ack_sha256="1" * 64,
            edge_send_evidence_sha256="2" * 64,
        )
        self.assertFalse(feedback_loss.penalizes_policy)
        self.assertIs(
            feedback_loss.learning_disposition,
            p2.ReconciliationLearningDisposition.CENSORED_NO_POLICY_PENALTY,
        )

        ue_fault = self._base_reconciliation(
            p2.TimeoutVerdict.UE_CONTROL_INFRASTRUCTURE_FAULT_EXCLUDED,
            edge_quality_ack_sha256="1" * 64,
            edge_send_evidence_sha256="2" * 64,
            ue_receive_ledger_sha256="3" * 64,
            packet_capture_evidence_sha256="4" * 64,
        )
        self.assertIs(
            ue_fault.learning_disposition,
            p2.ReconciliationLearningDisposition.EXCLUDED_INFRASTRUCTURE_FAULT,
        )

        evaluator_fault = self._base_reconciliation(
            p2.TimeoutVerdict.EVALUATOR_INFRASTRUCTURE_FAULT_EXCLUDED,
            infrastructure_evidence_sha256="5" * 64,
        )
        self.assertFalse(evaluator_fault.penalizes_policy)

        unresolved = self._base_reconciliation(
            p2.TimeoutVerdict.UNRESOLVED_CENSORING
        )
        self.assertFalse(unresolved.penalizes_policy)

        with self.assertRaises(p2.TimeoutReconciliationError):
            self._base_reconciliation(
                p2.TimeoutVerdict.ACTION_PATH_SERVICE_FAILURE_NEGATIVE
            )
        unverified = p2.ServiceTerminalEvidenceV2(
            completed_ticket=ticket,
            request=request,
            stage=p2.ServiceFailureStage.FEATURE_REASSEMBLY,
            failure_code="FEATURE_REASSEMBLY_EXPIRED",
            detected_at_ns=T0 + 120_000_000,
            terminal_document_sha256=hashlib.sha256(service_document).hexdigest(),
        )
        with self.assertRaises(p2.TimeoutReconciliationError):
            p2.TimeoutReconciliationV1(
                completed_ticket=ticket,
                verdict=p2.TimeoutVerdict.ACTION_PATH_SERVICE_FAILURE_NEGATIVE,
                reconciler_source_sha256=RECONCILER_SHA,
                reconciled_at_wall_ns=9_000,
                service_terminal_evidence=unverified,
            )
        with self.assertRaises(p2.TimeoutReconciliationError):
            p2.ServiceTerminalEvidenceV2._verify_from_reviewed_terminal(
                verifier_capability=object(),
                completed_ticket=ticket,
                request=request,
                stage=p2.ServiceFailureStage.FEATURE_REASSEMBLY,
                failure_code="FEATURE_REASSEMBLY_EXPIRED",
                detected_at_ns=T0 + 120_000_000,
                terminal_document=service_document,
            )
        with self.assertRaises(p2.TimeoutReconciliationError):
            self._base_reconciliation("FEEDBACK_ONLY_LOSS_CENSORED")

    def test_late_exact_quality_is_bound_to_original_ticket_and_censored(self) -> None:
        ticket, opened = self._timed_out_ticket()
        request = p2.DynamicFeatureRequestV2(
            run_id="run-v2",
            cell_id="cell-v2",
            stream_id="ue-a",
            controller_lineage_uuid=LINEAGE,
            capture_timestamp_ns=123_456_789,
            policy_decision_trace_sha256=TRACE_SHA,
            envelope=opened.envelope,
        )
        ack = self._ack(request)
        record = p2.TimeoutReconciliationV1(
            completed_ticket=ticket,
            verdict=p2.TimeoutVerdict.LATE_EXACT_QUALITY_CENSORED,
            reconciler_source_sha256=RECONCILER_SHA,
            reconciled_at_wall_ns=9_000,
            late_ack=ack,
            late_received_ns=ticket.deadline_ns + 1,
        )
        self.assertFalse(record.penalizes_policy)
        self.assertEqual(
            record.to_canonical_dict()["edge_quality_ack_sha256"],
            ack.canonical_sha256(),
        )
        with self.assertRaises(p2.TimeoutReconciliationError):
            replace(record, late_received_ns=ticket.deadline_ns)

        other_ticket, _ = self._timed_out_ticket(
            session_uuid=OTHER_SESSION,
            decision_seq=9,
            first_tensor_seq=30,
            first_frame_id=500,
        )
        with self.assertRaises(p2.TimeoutReconciliationError):
            replace(record, completed_ticket=other_ticket)

        wrong_lineage_request = replace(
            request,
            controller_lineage_uuid="11111111-2222-4333-8444-555555555555",
        )
        wrong_lineage_ack = self._ack(wrong_lineage_request)
        with self.assertRaises(p2.TimeoutReconciliationError):
            replace(record, late_ack=wrong_lineage_ack)

        wrong_trace_request = replace(
            request, policy_decision_trace_sha256="9" * 64
        )
        wrong_trace_ack = self._ack(wrong_trace_request)
        with self.assertRaises(p2.TimeoutReconciliationError):
            replace(record, late_ack=wrong_trace_ack)

    def test_non_timeout_ticket_and_unattested_ticket_are_rejected(self) -> None:
        action = self._action()
        controller = rtc.RewardTicketController(
            SESSION, controller_lineage_uuid=LINEAGE
        )
        controller.open_decision(
            decision_seq=7,
            tensor_seq=19,
            carla_frame_id=401,
            action=action,
            now_ns=T0,
            policy_decision_trace_sha256=TRACE_SHA,
        )
        controller.reuse_held_action(
            tensor_seq=20,
            carla_frame_id=402,
            now_ns=T0 + 50_000_000,
        )
        message = rtc.RewardFeedbackMessage(
            identity=ti.RewardFeedbackIdentity(
                session_uuid=SESSION,
                decision_seq=7,
                reward_tensor_seq=19,
                carla_frame_id=401,
                action=action,
            ),
            terminal_status=rtc.FeedbackTerminalStatus.REWARD_FINAL,
        )
        outcome = controller.submit_feedback(message, T0 + 80_000_000)
        ticket = outcome.completed_ticket
        assert ticket is not None
        with self.assertRaises(p2.TimeoutReconciliationError):
            p2.TimeoutReconciliationV1(
                completed_ticket=ticket,
                verdict=p2.TimeoutVerdict.UNRESOLVED_CENSORING,
                reconciler_source_sha256=RECONCILER_SHA,
                reconciled_at_wall_ns=9_000,
            )
        timeout, _ = self._timed_out_ticket()
        unattested = replace(timeout, _lineage_attestation=None)
        with self.assertRaises(p2.TimeoutReconciliationError):
            p2.TimeoutReconciliationV1(
                completed_ticket=unattested,
                verdict=p2.TimeoutVerdict.UNRESOLVED_CENSORING,
                reconciler_source_sha256=RECONCILER_SHA,
                reconciled_at_wall_ns=9_000,
            )


if __name__ == "__main__":
    unittest.main()

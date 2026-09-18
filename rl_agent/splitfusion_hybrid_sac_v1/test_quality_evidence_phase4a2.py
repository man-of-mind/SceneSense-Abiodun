"""Adversarial proofs for the Phase-4a.2 quality/evidence repair.

Pure deterministic unit tests: no CARLA, OAI, Docker, CUDA, filesystem evidence
or network service is used.  Extended detail rows are synthetic and do not
claim that the current live v1 producer emits the Phase-4a.2 support schema.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import unittest
from typing import Any, Dict, Mapping, Tuple

from . import state_reward_transition_contract as src
from .test_state_reward_transition_contract import BaseContractTest


class QualityEvidencePhase4a2Test(BaseContractTest):
    """The repaired reward path must fail closed at every evidence seam."""

    def _documents(self) -> Tuple[Any, src.EvaluationEligibilityV1, Dict[str, Any], Dict[str, Any]]:
        ticket = self._ticket()
        eligibility = self._eligibility()
        ack = self._ack_document(
            ticket.action,
            ticket=ticket,
            eligibility=eligibility,
            frame_id=ticket.reward_carla_frame_id,
        )
        detail = json.loads(
            json.dumps(self.quality_detail_documents[ack["dh"]])
        )
        return ticket, eligibility, ack, detail

    def _rebuild_ack(self, detail: Mapping[str, Any]) -> Dict[str, Any]:
        identity = {
            name: detail[name] for name in src.QUALITY_ACK_IDENTITY_FIELDS
        }
        return self.protocol.build_ack(
            identity_fields=identity,
            frozen_carla_frame_id=int(detail["frozen_carla_frame_id"]),
            timing=detail["timing"],
            quality=detail["quality"],
            evaluator_mode=str(detail["evaluator_mode"]),
            detail_sha256=self.protocol.detail_digest(detail),
        )

    def _bind(
        self,
        ticket: Any,
        eligibility: src.EvaluationEligibilityV1,
        ack: Mapping[str, Any],
        detail: Mapping[str, Any],
    ) -> src.QualityAckBindingV1:
        return src.QualityAckBindingV1.from_ack_document(
            ack,
            detail_document=detail,
            obligation=self._obligation(ticket),
            completed_ticket=ticket,
            eligibility=eligibility,
        )

    def test_legacy_v1_detail_without_support_extension_fails_closed(self) -> None:
        ticket, eligibility, _, detail = self._documents()
        detail.pop(src.QUALITY_DETAIL_SUPPORT_KEY)
        ack = self._rebuild_ack(detail)
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("unsupported", str(caught.exception))

    def test_actual_detail_digest_must_equal_ack_dh(self) -> None:
        ticket, eligibility, ack, detail = self._documents()
        detail["evaluator_mode"] = "tampered_after_ack"
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("does not match ACK dh", str(caught.exception))

    def test_actor_set_digest_and_recall_are_recomputed(self) -> None:
        ticket, eligibility, _, detail = self._documents()
        support = detail[src.QUALITY_DETAIL_SUPPORT_KEY]
        support["eligible_person_actor_ids"].append(9999)
        ack = self._rebuild_ack(detail)
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("not derived from gt_actor_rows", str(caught.exception))

        # Same cardinality plus a freshly recomputed actor-set digest was the
        # subtle bypass in the previous repair.  It must still fail because the
        # set itself is now derived from the raw actor rows.
        ticket, eligibility, _, detail = self._documents()
        support = detail[src.QUALITY_DETAIL_SUPPORT_KEY]
        forged_ids = [9001, 9002]
        support["eligible_person_actor_ids"] = forged_ids
        support["eligible_person_actor_ids_sha256"] = src.canonical_sha256(
            src.EvaluationEligibilityResultV1._actor_set_document(
                "person", tuple(forged_ids)
            )
        )
        ack = self._rebuild_ack(detail)
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("not derived from gt_actor_rows", str(caught.exception))

        ticket, eligibility, _, detail = self._documents()
        detail["quality"]["localization"]["person"]["recall"] = 0.75
        ack = self._rebuild_ack(detail)
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("person_recall", str(caught.exception))

    def test_mask_counts_reproduce_ack_iou(self) -> None:
        ticket, eligibility, _, detail = self._documents()
        detail["quality"]["segmentation"]["miou_person_iou"] = 0.5
        ack = self._rebuild_ack(detail)
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("cannot be reproduced", str(caught.exception))

        ticket, eligibility, _, detail = self._documents()
        support = detail[src.QUALITY_DETAIL_SUPPORT_KEY]
        support["seg_vehicle_union_pixels"] += 1
        ack = self._rebuild_ack(detail)
        with self.assertRaises(src.QualityContractError) as caught:
            self._bind(ticket, eligibility, ack, detail)
        self.assertIn("union", str(caught.exception))

    def test_wire_ack_is_idempotent_only_for_same_ticket(self) -> None:
        ticket, eligibility, ack, detail = self._documents()
        first = self._bind(ticket, eligibility, ack, detail)
        second = self._bind(ticket, eligibility, ack, detail)
        self.assertEqual(first.canonical_sha256(), second.canonical_sha256())

        # The reverse direction is equally important: the same ticket cannot
        # accept a second, different but internally self-consistent ACK.
        conflicting_detail = json.loads(json.dumps(detail))
        conflicting_detail["evaluator_mode"] = "conflicting_second_evaluation"
        conflicting_ack = self._rebuild_ack(conflicting_detail)
        self.assertNotEqual(
            self.protocol.digest(conflicting_ack),
            self.protocol.digest(ack),
        )
        with self.assertRaises(src.QualityAckReuseError):
            self._bind(
                ticket,
                eligibility,
                conflicting_ack,
                conflicting_detail,
            )

        # Same v1 wire identity, but a different hold manifest/ticket.  The v1
        # document cannot distinguish it, so the explicit registry must.
        other = self._ticket(extra_reuses=1)
        self.assertEqual(
            self._obligation(ticket).canonical_sha256(),
            self._obligation(other).canonical_sha256(),
        )
        self.assertNotEqual(ticket.canonical_sha256(), other.canonical_sha256())
        with self.assertRaises(src.QualityAckReuseError):
            src.QualityAckBindingV1.from_ack_document(
                ack,
                detail_document=detail,
                obligation=self._obligation(other),
                completed_ticket=other,
                eligibility=eligibility,
            )

        # A caller-created empty registry must not provide an escape hatch:
        # the verifier no longer accepts any registry parameter.
        with self.assertRaises(TypeError):
            src.QualityAckBindingV1.from_ack_document(
                ack,
                detail_document=detail,
                obligation=self._obligation(other),
                completed_ticket=other,
                eligibility=eligibility,
                use_registry=src.QualityAckUseRegistryV1("bypass-attempt"),
            )

    def test_quality_components_are_factory_only_and_have_no_loose_counts(self) -> None:
        ticket = self._ticket()
        honest = self._components(ticket)
        self.assertTrue(honest.is_attested)
        parameters = tuple(
            inspect.signature(src.QualityComponentsV1.from_ack_binding).parameters
        )
        self.assertEqual(parameters, ("evidence",))

        forged = dataclasses.replace(honest, _attestation=None)
        self.assertFalse(forged.is_attested)
        with self.assertRaises(src.UnattestedRecordError):
            forged.to_canonical_dict()
        with self.assertRaises(src.UnattestedRecordError):
            self._reward_spec().evaluate_quality(forged)

    def test_eligibility_hash_is_derived_from_semantics(self) -> None:
        valid = self._eligibility()
        self.assertEqual(
            valid.eligibility_contract_sha256,
            src.canonical_sha256(
                src.EvaluationEligibilityV1.contract_document(
                    eligibility_contract_id=valid.eligibility_contract_id,
                    max_range_m=valid.max_range_m,
                    fov_deg=valid.fov_deg,
                    visibility_rule=valid.visibility_rule,
                    segmentation_eligibility_masked=(
                        valid.segmentation_eligibility_masked
                    ),
                    reward_scope=valid.reward_scope,
                )
            ),
        )
        with self.assertRaises(src.QualityContractError):
            dataclasses.replace(valid, eligibility_contract_sha256="f" * 64)

    def test_fixture_support_cannot_enter_learning_and_action_is_exact(self) -> None:
        ticket = self._ticket()
        obligation = self._obligation(ticket)
        binding = self._binding(ticket, obligation=obligation)
        evidence = self._evidence(
            ticket, obligation=obligation, binding=binding
        )
        components = src.QualityComponentsV1.from_ack_binding(evidence)

        self.assertFalse(binding.eligibility_result.learning_ready)
        self.assertIs(
            binding.eligibility_result.producer_status,
            src.QualityProducerStatus.CONTRACT_FIXTURE_UNVERIFIED_SOURCE,
        )
        self.assertFalse(evidence.learning_ready)
        # Formula tests remain possible, but no scalar decision/replay reward
        # may be minted from caller-supplied synthetic source summaries.
        self.assertGreater(
            self._reward_spec().evaluate_quality(components).q_perc, 0.0
        )
        with self.assertRaises(src.QualityProducerUnavailableError):
            src.evaluate_completed_decision(
                ticket,
                self._reward_spec(),
                quality_components=components,
            )

        with self.assertRaises(src.QualityContractError) as caught:
            src.QualityEvidenceV1(
                gt_source=src.GroundTruthSource.CARLA_GT_EXACT,
                kind=src.EvidenceKind.PER_FRAME_CAUSAL_ACK,
                granularity=src.EvidenceGranularity.SINGLE_FRAME,
                eligibility=self._eligibility(),
                executed_action_sha256="f" * 64,
                gt_source_detail="forged action binding",
                ack_binding=binding,
            )
        self.assertIn("executed_action_sha256 differs", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Outcome-free tests for the frozen Run-4 analysis contract."""

from __future__ import annotations

import dataclasses
import math
import unittest

from . import contract as C


class AuthorityAndDesignTests(unittest.TestCase):
    def test_frozen_authorities_verify(self) -> None:
        result = C.verify_authorities()
        self.assertTrue(result["verified"])
        self.assertEqual(result["runner_commit"], C.RUNNER_COMMIT)

    def test_d2_population_is_exact(self) -> None:
        cycles = C.primary_cycle_indices()
        self.assertEqual(len(cycles), 224)
        self.assertEqual(cycles[0], (0, 1, 2))
        self.assertEqual(cycles[-1], (446, 447, 448))
        self.assertEqual(C.unclosed_decision_indices(), (448,))
        self.assertEqual(len(cycles) * C.EXPECTED_CELLS,
                         C.EXPECTED_PRIMARY_CYCLES)

    def test_profile_is_not_a_model_input(self) -> None:
        self.assertNotIn("profile_id", C.PRIMARY_MODEL_INPUT_FIELDS)
        self.assertIn("profile_id", C.AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS)

    def test_gate_thresholds_are_frozen(self) -> None:
        by_key = {gate.key: gate for gate in C.GATES}
        self.assertEqual(by_key["VALIDATION_NEXT_BACKLOG_ERROR"].thresholds,
                         {"max_nmae": 0.10,
                          "min_improvement_over_persistence": 0.20})
        self.assertEqual(
            by_key["VALIDATION_QUEUE_CLEARANCE_LATENCY_ERROR"].thresholds,
            {"max_p50_error_ms": 17.0, "max_p95_error_ms": 34.0})
        self.assertEqual(
            by_key["VALIDATION_QUEUE_CLEARANCE_OUTCOME"].thresholds["max_brier"],
            0.15)


class PacketizationTests(unittest.TestCase):
    def test_production_boundary_cases(self) -> None:
        self.assertEqual(C.production_datagram_count(12_492), 1)
        self.assertEqual(C.production_udp_application_bytes(12_492), 12_500)
        self.assertEqual(C.production_datagram_count(12_493), 2)
        self.assertEqual(C.production_udp_application_bytes(12_493), 12_509)
        self.assertEqual(C.production_unfragmented_ipv4_baseline_bytes(12_493),
                         12_565)
        self.assertGreater(C.PRODUCTION_UNFRAGMENTED_IPV4_BYTES,
                           C.REGISTERED_PATH_MTU_BYTES)

    def test_calibration_and_production_headers_are_not_conflated(self) -> None:
        value = 12_493
        self.assertNotEqual(C.calibration_socket_bytes(value),
                            C.production_udp_application_bytes(value))


class ScalingAndFreshnessTests(unittest.TestCase):
    def test_scale_is_fit_only_nearest_rank_and_unclipped(self) -> None:
        rows = [C.BacklogScaleObservation(
            cell_id="fit", cycle_start_index=2 * i, partition=C.FIT,
            pre_action_backlog_bytes=i) for i in range(1, 101)]
        result = C.derive_backlog_log1p_scale(rows)
        self.assertEqual(result.nearest_rank_one_based, 99)
        self.assertEqual(result.population_count, 100)
        self.assertAlmostEqual(result.backlog_log1p_scale, math.log1p(99))
        self.assertFalse(result.clipping)
        self.assertGreater(C.normalize_backlog_unclipped(1_000_000,
                           scale=result.backlog_log1p_scale), 1.0)

    def test_validation_cannot_enter_scale(self) -> None:
        row = C.BacklogScaleObservation(
            "validation", 0, C.VALIDATION, 10)
        with self.assertRaisesRegex(C.ContractError, "FIT cells only"):
            C.derive_backlog_log1p_scale([row])

    def test_all_zero_scale_refuses(self) -> None:
        row = C.BacklogScaleObservation("fit", 0, C.FIT, 0)
        with self.assertRaisesRegex(C.ContractError, "finite and positive"):
            C.derive_backlog_log1p_scale([row])

    def test_freshness_accepts_boundary_and_never_zero_fills(self) -> None:
        ages = {field: C.FRESHNESS_MAX_AGE_NS
                for field in C.FRESHNESS_POLICY_FIELDS}
        self.assertTrue(C.evaluate_freshness(ages).accepted)
        ages["camera_si"] = None
        missing = C.evaluate_freshness(ages)
        self.assertFalse(missing.accepted)
        self.assertEqual(missing.outcome, C.FRESHNESS_FALLBACK)
        self.assertEqual(missing.missing, ("camera_si",))
        ages["camera_si"] = C.FRESHNESS_MAX_AGE_NS + 1
        stale = C.evaluate_freshness(ages)
        self.assertEqual(stale.stale, ("camera_si",))


def request(**overrides: object) -> C.QueueServiceRequestV1:
    values: dict[str, object] = {
        "model_binding_sha256": "a" * 64,
        "duration_steps": 2,
        "step_period_ns": 100_000_000,
        "pre_action_backlog_bytes": 10_000,
        "backlog_age_ns": 10_000_000,
        "prior_new_data_round0_table0_ul_mcs": 12,
        "mcs_age_ns": 20_000_000,
        "decision_mode_id": 6,
        "decision_q_e4": 5_000,
        "decision_total_transmitted_bytes": 374_264,
        "held_mode_id": 6,
        "held_q_e4": 5_000,
        "held_total_transmitted_bytes": 374_264,
    }
    values.update(overrides)
    return C.QueueServiceRequestV1(**values)  # type: ignore[arg-type]


class ProviderSchemaTests(unittest.TestCase):
    def test_request_has_no_profile_and_requires_exact_held_action(self) -> None:
        self.assertNotIn("profile_id",
                         C.QueueServiceRequestV1.__dataclass_fields__)
        request().validate()
        with self.assertRaisesRegex(C.ContractError, "reuse the exact"):
            request(held_q_e4=7_000).validate()

    def test_stale_mcs_or_backlog_refuses(self) -> None:
        with self.assertRaisesRegex(C.ContractError, "backlog is stale"):
            request(backlog_age_ns=C.FRESHNESS_MAX_AGE_NS + 1).validate()
        with self.assertRaisesRegex(C.ContractError, "MCS is stale"):
            request(mcs_age_ns=C.FRESHNESS_MAX_AGE_NS + 1).validate()

    def test_support_disclosures_and_range_refusal(self) -> None:
        measured = request()
        self.assertNotIn(C.MODE_TRANSFER_DISCLOSURE,
                         C.support_disclosures(measured))
        transferred = request(decision_mode_id=0, held_mode_id=0,
                              decision_total_transmitted_bytes=200_000,
                              held_total_transmitted_bytes=200_000)
        disclosures = C.support_disclosures(transferred)
        self.assertIn(C.MODE_TRANSFER_DISCLOSURE, disclosures)
        self.assertIn(C.PAYLOAD_INTERPOLATION_DISCLOSURE, disclosures)
        with self.assertRaisesRegex(C.ContractError,
                                    C.PAYLOAD_OUT_OF_RANGE_REFUSAL):
            C.require_payload_support(request(
                decision_total_transmitted_bytes=1_000,
                held_total_transmitted_bytes=1_000))

    def test_distribution_uses_exact_external_draw(self) -> None:
        atoms = (
            C.JointServiceAtomV1(10, 20, 30, 40, 2),
            C.JointServiceAtomV1(11, 21, 31, None, 1),
        )
        dist = C.QueueServiceDistributionV1(
            "a" * 64, "b" * 64, "IN_SUPPORT",
            (C.QUEUE_ONLY_DISCLOSURE, C.SINGLE_UE_DISCLOSURE,
             C.PROFILE_TRANSFER_DISCLOSURE,
             C.PRODUCTION_PACKETIZATION_STATUS,
             C.BYTE_DOMAIN_CONVERSION_PROVEN), atoms)
        self.assertIs(dist.exact_draw(0), atoms[0])
        self.assertIs(dist.exact_draw(1), atoms[0])
        self.assertIs(dist.exact_draw(2), atoms[1])
        self.assertEqual(dist.clearance_probability_fraction(40), (2, 3))

    def test_model_binding_seals_fit_and_validation_separately(self) -> None:
        values = {
            field: f"{index:x}" * 64
            for index, field in enumerate((
                "raw_manifest_sha256", "canonical_d2_cycles_sha256",
                "fit_cell_set_sha256", "fit_rows_sha256",
                "validation_cell_set_sha256", "validation_rows_sha256",
                "service_table_sha256", "byte_domain_proof_sha256",
                "backlog_scale_sha256",
                "freshness_policy_sha256", "gate_report_sha256",
                "verifier_report_sha256"), start=1)
        }
        binding = C.QueueServiceModelBindingV1(
            **values, validation_influenced_fit=False,
            profile_is_model_input=False,
            packetization_transfer_used_for_latency=False,
            byte_domain_conversion_verified=True)
        self.assertEqual(len(binding.model_binding_sha256), 64)
        bad = dataclasses.replace(binding, validation_influenced_fit=True)
        with self.assertRaisesRegex(C.ContractError, "validation influenced"):
            bad.validate()

    def test_byte_domain_is_unresolved_until_exact_proof(self) -> None:
        unresolved = C.support_disclosures(request())
        self.assertIn(C.BYTE_DOMAIN_TRANSFER_UNRESOLVED, unresolved)
        with self.assertRaisesRegex(C.ContractError,
                                    "mandatory scope disclosures"):
            C.QueueServiceDistributionV1(
                "a" * 64, "b" * 64, "IN_SUPPORT", unresolved,
                (C.JointServiceAtomV1(1, 1, 0, 1, 1),)).validate()

        proof = C.ByteDomainProofV1(
            **{field: "a" * 64 for field in C.BYTE_DOMAIN_REQUIRED_PROOFS},
            exact_per_decision_accounting=True,
            no_calibration_chunk_effect_as_action_effect=True,
            production_fragmentation_observed_not_assumed=True,
            packetization_delivery_and_latency_excluded=True)
        qualified = C.qualified_support_disclosures(request(), proof)
        self.assertNotIn(C.BYTE_DOMAIN_TRANSFER_UNRESOLVED, qualified)
        self.assertIn(C.BYTE_DOMAIN_CONVERSION_PROVEN, qualified)


class CouplingBoundaryTests(unittest.TestCase):
    @staticmethod
    def _byte_proof() -> C.ByteDomainProofV1:
        return C.ByteDomainProofV1(
            **{field: "a" * 64 for field in C.BYTE_DOMAIN_REQUIRED_PROOFS},
            exact_per_decision_accounting=True,
            no_calibration_chunk_effect_as_action_effect=True,
            production_fragmentation_observed_not_assumed=True,
            packetization_delivery_and_latency_excluded=True)

    def test_no_proof_is_state_only_and_blocks_training(self) -> None:
        result = C.resolve_composite_coupling(None)
        self.assertEqual(result.alternative, C.STATE_TRANSITION_ONLY)
        self.assertFalse(result.training_ready)

    def test_complete_proof_enables_residualized_coupling(self) -> None:
        byte_proof = self._byte_proof()
        proof = C.ResidualizationProofV1(
            **{field: (byte_proof.proof_sha256
                       if field == "byte_domain_proof_sha256"
                       else str(i + 1) * 64)
               for i, field in enumerate(C.RESIDUALIZATION_REQUIRED_PROOFS)},
            exact_no_double_count=True,
            service_replaces_overlapping_transport_segment=True,
            all_residual_intervals_nonnegative=True,
            receiver_delivery_population_unchanged=True)
        result = C.resolve_composite_coupling(
            proof, byte_domain_proof=byte_proof)
        self.assertEqual(result.alternative,
                         C.RESIDUALIZED_SERVICE_TO_LATENCY)
        self.assertTrue(result.training_ready)

    def test_partial_proof_does_not_silently_downgrade(self) -> None:
        proof = C.ResidualizationProofV1(
            **{field: "a" * 64
               for field in C.RESIDUALIZATION_REQUIRED_PROOFS},
            exact_no_double_count=False,
            service_replaces_overlapping_transport_segment=True,
            all_residual_intervals_nonnegative=True,
            receiver_delivery_population_unchanged=True)
        with self.assertRaisesRegex(C.ContractError, "no-double-count"):
            C.resolve_composite_coupling(
                proof, byte_domain_proof=self._byte_proof())

    def test_residualized_path_refuses_missing_byte_domain_proof(self) -> None:
        proof = C.ResidualizationProofV1(
            **{field: "a" * 64
               for field in C.RESIDUALIZATION_REQUIRED_PROOFS},
            exact_no_double_count=True,
            service_replaces_overlapping_transport_segment=True,
            all_residual_intervals_nonnegative=True,
            receiver_delivery_population_unchanged=True)
        with self.assertRaisesRegex(C.ContractError, "typed byte-domain"):
            C.resolve_composite_coupling(proof)

    def test_contract_document_is_stable_and_explicit(self) -> None:
        document = C.contract_document()
        self.assertEqual(C.CONTRACT_SHA256, C.canonical_sha256(document))
        self.assertEqual(document["coupling"]["blind_add_to_288_total"],
                         "FORBIDDEN")
        self.assertFalse(document["coupling"][
            "alternative_a_training_ready"])


if __name__ == "__main__":
    unittest.main()

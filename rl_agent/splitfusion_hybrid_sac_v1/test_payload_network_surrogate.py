"""Deterministic tests for the strict offline payload/network surrogate."""

from __future__ import annotations

import io
import json
import math
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from . import payload_network_surrogate as ns
from .action_contract import CATALOG_SHA256
from .anchor_store import (
    ACTION_SUMMARY_SHA256,
    NETWORK_PROFILE_ORDER,
    PROFILE_LATENCY_SHA256,
)


class PayloadNetworkSurrogateTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = ns.build_payload_network_surrogate()


class EvidenceBindingTest(PayloadNetworkSurrogateTestBase):
    def test_exact_source_bindings_and_inventory(self) -> None:
        model = self.model
        self.assertEqual(model.source_action_summary_sha256, ACTION_SUMMARY_SHA256)
        self.assertEqual(
            model.source_profile_latency_sha256, PROFILE_LATENCY_SHA256
        )
        self.assertEqual(model.catalog_sha256, CATALOG_SHA256)
        self.assertEqual(
            model.surrogate_implementation_sha256,
            ns._current_implementation_sha256(model.contract),
        )
        self.assertEqual(len(model.observations), 288)
        self.assertEqual(
            {row.network_profile for row in model.observations},
            set(NETWORK_PROFILE_ORDER),
        )
        self.assertEqual({row.mode_id for row in model.observations}, set(range(12)))

    def test_latency_missingness_is_not_zero_imputed(self) -> None:
        missing = [row for row in self.model.observations if row.network_support == 0]
        observed = [row for row in self.model.observations if row.network_support > 0]
        self.assertEqual(len(missing), 66)
        self.assertEqual(len(observed), 222)
        for row in missing:
            self.assertIsNone(row.network_p50_ms)
            self.assertIsNone(row.network_p95_ms)
            self.assertIsNone(row.network_p99_ms)
        for row in observed:
            self.assertGreater(row.network_p50_ms, 0.0)
            self.assertGreaterEqual(row.network_p95_ms, row.network_p50_ms)
            self.assertGreaterEqual(row.network_p99_ms, row.network_p95_ms)

        low_support = [
            row
            for row in observed
            if row.network_support < ns.LATENCY_MIN_SUPPORT
        ]
        qualified = [
            row
            for row in observed
            if row.network_support >= ns.LATENCY_MIN_SUPPORT
        ]
        self.assertEqual(len(low_support), 15)
        self.assertEqual(len(qualified), 207)

    def test_retained_latency_population_is_edge_admitted_survivors(self) -> None:
        self.assertTrue(
            all(
                row.network_support <= row.edge_admissions
                for row in self.model.observations
            )
        )
        self.assertEqual(
            sum(
                row.network_support == row.edge_admissions
                for row in self.model.observations
            ),
            46,
        )

    def test_datagram_count_is_a_real_bound_support_coordinate(self) -> None:
        for row in self.model.observations:
            nominal = math.ceil(
                row.payload_bytes / ns.UDP_PAYLOAD_CAPACITY_BYTES
            )
            self.assertEqual(row.datagram_count, nominal)

    def test_expectation_accessor_does_not_claim_an_observed_event(self) -> None:
        row = next(
            row
            for row in self.model.observations
            if row.network_support >= ns.LATENCY_MIN_SUPPORT
        )
        prediction = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        latency = prediction.conditional_retained_survivor_latency_model()
        self.assertGreater(latency.p50_ms, 0.0)
        with self.assertRaises(ns.ExtrapolationRefusedError):
            prediction.require_latency(
                edge_admission_succeeded=False,
                downstream_result_retained=False,
            )

    def test_prevalidated_session_matches_fail_closed_public_prediction(self) -> None:
        row = self.model.observations[0]
        session = self.model.prevalidated_prediction_session()
        public = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        hot = session.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        self.assertEqual(public.to_canonical_dict(), hot.to_canonical_dict())
        self.assertEqual(session.model_sha256, self.model.canonical_sha256())

    def test_preflight_states_the_scientific_boundaries(self) -> None:
        document = self.model.preflight_document()
        self.assertEqual(document["evidence_class"], ns.EVIDENCE_CLASS)
        self.assertEqual(document["inventory"]["cells"], 288)
        self.assertEqual(document["inventory"]["latency_observed_cells"], 222)
        self.assertEqual(
            document["inventory"]["zero_latency_support_cells_excluded"], 66
        )
        self.assertFalse(
            document["model_scope"]["network_profile_is_policy_state"]
        )
        self.assertEqual(document["contract"]["extrapolation"], "FAIL_CLOSED")
        self.assertEqual(
            document["contract"]["udp_datagram_bytes_including_header"],
            12_500,
        )
        self.assertEqual(document["contract"]["udp_chunk_header_bytes"], 8)
        self.assertEqual(
            document["contract"]["udp_payload_capacity_bytes"], 12_492
        )
        self.assertEqual(document["contract"]["latency_min_support"], 100)
        self.assertIn(
            "EDGE_ADMITTED_AND_DOWNSTREAM_RESULT_RETAINED",
            document["model_scope"]["latency_selection_disclosure"],
        )
        self.assertTrue(
            document["model_scope"][
                "latency_observation_is_post_admission_selected"
            ]
        )
        self.assertFalse(
            document["model_scope"][
                "causal_pre_admission_arrival_latency_available"
            ]
        )
        self.assertFalse(document["model_scope"]["per_frame_csi_conditioning"])
        self.assertFalse(
            document["model_scope"]["temporal_channel_correlation_modeled"]
        )
        self.assertEqual(document["inventory"]["latency_qualified_cells"], 207)
        self.assertEqual(
            document["source_binding"]["production_transport_sha256"],
            ns.PRODUCTION_TRANSPORT_SHA256,
        )
        self.assertEqual(
            document["source_binding"]["production_runtime_contract_sha256"],
            ns.PRODUCTION_RUNTIME_CONTRACT_SHA256,
        )
        self.assertEqual(
            document["source_binding"]["source_analysis_builder_sha256"],
            ns.SOURCE_ANALYSIS_BUILDER_SHA256,
        )
        self.assertEqual(
            document["source_binding"]["source_analysis_summary_sha256"],
            ns.SOURCE_ANALYSIS_SUMMARY_SHA256,
        )
        self.assertEqual(
            document["source_binding"]["source_analysis_manifest_sha256"],
            ns.SOURCE_ANALYSIS_MANIFEST_SHA256,
        )


class MonotoneFitTest(PayloadNetworkSurrogateTestBase):
    def test_rate_curves_are_nonincreasing_in_payload(self) -> None:
        for profile, fitted in self.model.profile_models.items():
            del profile
            for curve in (fitted.reassembly_curve, fitted.admission_curve):
                values = [block.value for block in curve.blocks]
                self.assertTrue(
                    all(left >= right for left, right in zip(values, values[1:]))
                )

    def test_latency_curves_are_nondecreasing_in_payload(self) -> None:
        for fitted in self.model.profile_models.values():
            for curve in fitted.latency_curves.values():
                values = [block.value for block in curve.blocks]
                self.assertTrue(
                    all(left <= right for left, right in zip(values, values[1:]))
                )

    def test_weighted_pava_pools_a_real_violation(self) -> None:
        curve = ns._MonotoneCurve.fit(
            [(1.0, 0.9, 10.0), (2.0, 0.4, 10.0), (3.0, 0.6, 20.0)],
            increasing=False,
        )
        self.assertEqual(len(curve.blocks), 2)
        self.assertAlmostEqual(curve.blocks[-1].value, (4.0 + 12.0) / 30.0)
        self.assertEqual(curve.blocks[-1].members, 2)

    def test_interpolation_does_not_break_monotonicity(self) -> None:
        curve = ns._MonotoneCurve.fit(
            [(1.0, 0.9, 1.0), (2.0, 0.7, 1.0), (3.0, 0.2, 1.0)],
            increasing=False,
        )
        values = [curve.predict(1.0 + index / 20.0).value for index in range(41)]
        self.assertTrue(
            all(left >= right for left, right in zip(values, values[1:]))
        )


class PredictionTest(PayloadNetworkSurrogateTestBase):
    def _small_observation(self, profile: str):
        return min(
            (
                row
                for row in self.model.observations
                if row.network_profile == profile and row.network_support > 0
            ),
            key=lambda row: row.payload_bytes,
        )

    def test_prediction_is_coherent_and_labelled_modeled(self) -> None:
        row = self._small_observation("ADVERSE_STABLE")
        prediction = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        self.assertEqual(prediction.evidence_class, ns.EVIDENCE_CLASS)
        self.assertFalse(prediction.policy_observation_admissible)
        self.assertAlmostEqual(
            prediction.p_edge_admission_given_sent,
            prediction.p_complete_reassembly_given_sent
            * prediction.p_edge_admission_given_reassembled,
            places=15,
        )
        for value in (
            prediction.p_complete_reassembly_given_sent,
            prediction.p_edge_admission_given_reassembled,
            prediction.p_edge_admission_given_sent,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        with self.assertRaises(ns.ExtrapolationRefusedError):
            prediction.require_latency(
                edge_admission_succeeded=False,
                downstream_result_retained=True,
            )
        with self.assertRaises(ns.ExtrapolationRefusedError):
            prediction.require_latency(
                edge_admission_succeeded=True,
                downstream_result_retained=False,
            )
        latency = prediction.require_latency(
            edge_admission_succeeded=True,
            downstream_result_retained=True,
        )
        self.assertLessEqual(latency.p50_ms, latency.p95_ms)
        self.assertLessEqual(latency.p95_ms, latency.p99_ms)
        self.assertGreater(latency.effective_support, 0.0)

    def test_binomial_reference_intervals_contain_point_estimates(self) -> None:
        row = self._small_observation("FAVORABLE_STABLE")
        prediction = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        self.assertLessEqual(
            prediction.reassembly_binomial_reference_interval95[0],
            prediction.p_complete_reassembly_given_sent,
        )
        self.assertGreaterEqual(
            prediction.reassembly_binomial_reference_interval95[1],
            prediction.p_complete_reassembly_given_sent,
        )
        self.assertLessEqual(
            prediction.admission_binomial_reference_interval95[0],
            prediction.p_edge_admission_given_reassembled,
        )
        self.assertGreaterEqual(
            prediction.admission_binomial_reference_interval95[1],
            prediction.p_edge_admission_given_reassembled,
        )

    def test_one_extra_datagram_is_refused_not_tolerated(self) -> None:
        row = self._small_observation("FAVORABLE_STABLE")
        nominal = math.ceil(
            row.payload_bytes / ns.UDP_PAYLOAD_CAPACITY_BYTES
        )
        self.assertEqual(row.datagram_count, nominal)
        with self.assertRaises(ns.ExtrapolationRefusedError):
            self.model.predict(
                network_profile=row.network_profile,
                payload_bytes=row.payload_bytes,
                datagram_count=nominal + 1,
            )

    def test_exact_12492_byte_fragment_boundary(self) -> None:
        self.model.predict(
            network_profile="FAVORABLE_STABLE",
            payload_bytes=12_492,
            datagram_count=1,
        )
        self.model.predict(
            network_profile="FAVORABLE_STABLE",
            payload_bytes=12_493,
            datagram_count=2,
        )
        with self.assertRaises(ns.ExtrapolationRefusedError):
            self.model.predict(
                network_profile="FAVORABLE_STABLE",
                payload_bytes=12_493,
                datagram_count=1,
            )

    def test_delivery_supported_but_latency_unsupported_stays_missing(self) -> None:
        # Largest payload lies in the rate envelope but has no conditional
        # network timing; it must not become a zero-millisecond latency.
        row = max(
            (
                row
                for row in self.model.observations
                if row.network_profile == "ADVERSE_STABLE"
            ),
            key=lambda row: row.payload_bytes,
        )
        prediction = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        self.assertIsNone(prediction.to_canonical_dict()["latency"])
        self.assertFalse(prediction.support.latency_supported)
        with self.assertRaises(ns.ExtrapolationRefusedError):
            prediction.require_latency(
                edge_admission_succeeded=True,
                downstream_result_retained=True,
            )

    def test_low_support_cell_has_no_latency_prediction(self) -> None:
        checked = 0
        for row in self.model.observations:
            if not 0 < row.network_support < ns.LATENCY_MIN_SUPPORT:
                continue
            prediction = self.model.predict(
                network_profile=row.network_profile,
                payload_bytes=row.payload_bytes,
                datagram_count=row.datagram_count,
            )
            self.assertFalse(prediction.support.latency_supported)
            self.assertIsNone(prediction.to_canonical_dict()["latency"])
            if prediction.support.latency_effective_samples is not None:
                self.assertEqual(
                    prediction.support.latency_effective_samples,
                    float(row.network_support),
                )
            with self.assertRaises(ns.ExtrapolationRefusedError):
                prediction.require_latency(
                    edge_admission_succeeded=True,
                    downstream_result_retained=True,
                )
            checked += 1
        self.assertEqual(checked, 15)

    def test_every_qualified_exact_cell_exposes_survivor_latency(self) -> None:
        checked = 0
        for row in self.model.observations:
            if row.network_support < ns.LATENCY_MIN_SUPPORT:
                continue
            prediction = self.model.predict(
                network_profile=row.network_profile,
                payload_bytes=row.payload_bytes,
                datagram_count=row.datagram_count,
            )
            self.assertTrue(prediction.support.latency_supported)
            latency = prediction.require_latency(
                edge_admission_succeeded=True,
                downstream_result_retained=True,
            )
            self.assertGreaterEqual(
                latency.effective_support, ns.LATENCY_MIN_SUPPORT
            )
            checked += 1
        self.assertEqual(checked, 207)

    def test_payload_extrapolation_is_refused(self) -> None:
        with self.assertRaises(ns.ExtrapolationRefusedError):
            self.model.predict(
                network_profile="FAVORABLE_STABLE",
                payload_bytes=1.0,
                datagram_count=1,
            )

    def test_fragmentation_contradiction_is_refused(self) -> None:
        row = self._small_observation("FADE_RECOVERY")
        with self.assertRaises(ns.ExtrapolationRefusedError):
            self.model.predict(
                network_profile=row.network_profile,
                payload_bytes=row.payload_bytes,
                datagram_count=row.datagram_count + 4,
            )

    def test_unknown_profile_and_nonfinite_values_are_refused(self) -> None:
        for profile, payload in (
            ("UNSEEN_PROFILE", 10_000.0),
            ("FAVORABLE_STABLE", float("nan")),
            ("FAVORABLE_STABLE", float("inf")),
        ):
            with self.subTest(profile=profile, payload=payload):
                with self.assertRaises(ns.ExtrapolationRefusedError):
                    self.model.predict(
                        network_profile=profile,
                        payload_bytes=payload,
                        datagram_count=1,
                    )

    def test_profile_context_cannot_become_policy_state(self) -> None:
        row = self._small_observation("MID_VARIABLE")
        prediction = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        with self.assertRaises(ns.PrivilegedContextLeakError):
            prediction.as_policy_observation()

    def test_prediction_serialization_is_deterministic(self) -> None:
        row = self._small_observation("MID_VARIABLE")
        prediction = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        first = json.dumps(prediction.to_canonical_dict(), sort_keys=True)
        second = json.dumps(prediction.to_canonical_dict(), sort_keys=True)
        self.assertEqual(first, second)


class WholeModeValidationTest(PayloadNetworkSurrogateTestBase):
    def test_rate_validation_covers_every_cell_or_marks_unsupported(self) -> None:
        for name in (
            "reassembly_per_sent",
            "admission_given_reassembly",
            "admission_per_sent_chain",
        ):
            metric = self.model.validation[name]
            self.assertEqual(metric.evaluated + metric.unsupported, 288)

    def test_validation_is_close_to_the_independent_exploratory_check(self) -> None:
        reassembly = self.model.validation["reassembly_per_sent"]
        conditional = self.model.validation["admission_given_reassembly"]
        chained = self.model.validation["admission_per_sent_chain"]
        p50 = self.model.validation["network_p50_ms_support_ge_100"]

        # These are deliberately tolerances, not tuning targets. The current
        # implementation is weighted monotone and chains the two probabilities;
        # the earlier exploratory check used a slightly different interpolation.
        self.assertLess(reassembly.mae, 0.010)
        self.assertLess(reassembly.p90_absolute_error, 0.030)
        self.assertGreater(reassembly.r_squared, 0.99)
        self.assertLess(conditional.mae, 0.010)
        self.assertLess(conditional.p90_absolute_error, 0.030)
        self.assertGreater(conditional.r_squared, 0.99)
        self.assertLess(chained.mae, 0.011)
        self.assertGreater(chained.r_squared, 0.99)
        self.assertLess(p50.mae, 1.5)
        self.assertLess(p50.p90_absolute_error, 3.5)
        self.assertGreater(p50.r_squared, 0.98)

    def test_tail_percentiles_are_reported_not_hidden(self) -> None:
        p95 = self.model.validation["network_p95_ms_support_ge_100"]
        p99 = self.model.validation["network_p99_ms_support_ge_100"]
        self.assertGreater(p95.evaluated, 0)
        self.assertGreater(p99.evaluated, 0)
        self.assertGreater(p99.mae, p95.mae)

    def test_model_digest_is_stable_across_fresh_fits(self) -> None:
        another = ns.build_payload_network_surrogate()
        self.assertEqual(self.model.canonical_sha256(), another.canonical_sha256())


class DeepFreezeAndDigestClosureTest(PayloadNetworkSurrogateTestBase):
    @staticmethod
    def _read_with_one_source_edit(target: Path):
        original = Path.read_bytes

        def read_bytes(path: Path) -> bytes:
            raw = original(path)
            if path.resolve() == target.resolve():
                return raw + b"\n# adversarial source drift\n"
            return raw

        return read_bytes

    def test_all_nested_model_mappings_are_read_only(self) -> None:
        profile = self.model.profile_models["FAVORABLE_STABLE"]
        with self.assertRaises(TypeError):
            self.model.profile_models["NEW"] = profile
        with self.assertRaises(TypeError):
            self.model.validation["NEW"] = next(iter(self.model.validation.values()))
        with self.assertRaises(TypeError):
            profile.latency_curves["p50"] = profile.latency_curves["p95"]
        with self.assertRaises(TypeError):
            profile.held_mode_absolute_residual_p90_ms["p50"] = 0.0
        with self.assertRaises(TypeError):
            self.model.observations[0] = self.model.observations[1]

    def test_prediction_residual_mappings_are_deep_frozen(self) -> None:
        row = min(
            (
                row
                for row in self.model.observations
                if row.network_profile == "FAVORABLE_STABLE"
                and row.network_support >= ns.LATENCY_MIN_SUPPORT
            ),
            key=lambda item: item.payload_bytes,
        )
        latency = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        ).require_latency(
            edge_admission_succeeded=True,
            downstream_result_retained=True,
        )
        with self.assertRaises(TypeError):
            latency.held_mode_absolute_residual_p90_ms["p50"] = 0.0
        with self.assertRaises(TypeError):
            latency.held_mode_residual_band_ms["p50"] = (0.0, 0.0)

    def test_every_prediction_affecting_curve_value_changes_digest(self) -> None:
        profile_name = "FAVORABLE_STABLE"
        profile = self.model.profile_models[profile_name]
        curve = profile.reassembly_curve
        last = curve.blocks[-1]
        changed_last = replace(
            last,
            weighted_sum=last.weighted_sum + last.weight * 1e-4,
        )
        changed_curve = replace(
            curve, blocks=curve.blocks[:-1] + (changed_last,)
        )
        changed_profile = replace(profile, reassembly_curve=changed_curve)
        changed_profiles = dict(self.model.profile_models)
        changed_profiles[profile_name] = changed_profile
        changed_model = replace(
            self.model,
            profile_models=MappingProxyType(changed_profiles),
        )
        changed_model.revalidate()
        self.assertNotEqual(
            self.model.canonical_sha256(), changed_model.canonical_sha256()
        )

        row = max(
            (
                row
                for row in self.model.observations
                if row.network_profile == profile_name
            ),
            key=lambda item: item.payload_bytes,
        )
        original = self.model.predict(
            network_profile=profile_name,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        changed = changed_model.predict(
            network_profile=profile_name,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        )
        self.assertNotEqual(
            original.p_complete_reassembly_given_sent,
            changed.p_complete_reassembly_given_sent,
        )

    def test_residual_diagnostic_changes_digest_and_output_band(self) -> None:
        profile_name = "ADVERSE_STABLE"
        profile = self.model.profile_models[profile_name]
        residual = dict(profile.held_mode_absolute_residual_p90_ms)
        residual["p50"] += 1.0
        changed_profile = replace(
            profile,
            held_mode_absolute_residual_p90_ms=MappingProxyType(residual),
        )
        profiles = dict(self.model.profile_models)
        profiles[profile_name] = changed_profile
        changed_model = replace(
            self.model, profile_models=MappingProxyType(profiles)
        )
        changed_model.revalidate()
        self.assertNotEqual(
            self.model.canonical_sha256(), changed_model.canonical_sha256()
        )

        row = min(
            (
                row
                for row in self.model.observations
                if row.network_profile == profile_name
                and row.network_support >= ns.LATENCY_MIN_SUPPORT
            ),
            key=lambda item: item.payload_bytes,
        )
        before = self.model.predict(
            network_profile=profile_name,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        ).require_latency(
            edge_admission_succeeded=True,
            downstream_result_retained=True,
        )
        after = changed_model.predict(
            network_profile=profile_name,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        ).require_latency(
            edge_admission_succeeded=True,
            downstream_result_retained=True,
        )
        self.assertNotEqual(
            before.held_mode_residual_band_ms["p50"],
            after.held_mode_residual_band_ms["p50"],
        )

    def test_contract_or_source_tamper_fails_closed(self) -> None:
        bad_contract = replace(self.model.contract, latency_min_support=99)
        with self.assertRaises(ns.EvidenceDefinitionError):
            replace(self.model, contract=bad_contract).revalidate()
        with self.assertRaises(ns.EvidenceDefinitionError):
            replace(
                self.model, source_action_summary_sha256="0" * 64
            ).revalidate()

    def test_implementation_source_edit_changes_fresh_model_identity(self) -> None:
        target = (
            ns._project_root()
            / ns.SURROGATE_IMPLEMENTATION_RELATIVE_PATH
        )
        baseline = self.model.canonical_sha256()
        with patch.object(
            Path,
            "read_bytes",
            new=self._read_with_one_source_edit(target),
        ):
            changed = ns.build_payload_network_surrogate()
            self.assertNotEqual(
                changed.surrogate_implementation_sha256,
                self.model.surrogate_implementation_sha256,
            )
            self.assertNotEqual(changed.canonical_sha256(), baseline)

    def test_implementation_source_edit_invalidates_bound_model(self) -> None:
        target = (
            ns._project_root()
            / ns.SURROGATE_IMPLEMENTATION_RELATIVE_PATH
        )
        with patch.object(
            Path,
            "read_bytes",
            new=self._read_with_one_source_edit(target),
        ):
            with self.assertRaisesRegex(
                ns.EvidenceDefinitionError,
                "implementation source changed",
            ):
                self.model.revalidate()

    def test_semantic_source_edits_fail_binding(self) -> None:
        for relative_path in (
            ns.SOURCE_ANALYSIS_BUILDER_RELATIVE_PATH,
            ns.SOURCE_ANALYSIS_SUMMARY_RELATIVE_PATH,
            ns.SOURCE_ANALYSIS_MANIFEST_RELATIVE_PATH,
        ):
            target = ns._project_root() / relative_path
            with self.subTest(relative_path=relative_path):
                with patch.object(
                    Path,
                    "read_bytes",
                    new=self._read_with_one_source_edit(target),
                ):
                    with self.assertRaises(ns.EvidenceDefinitionError):
                        ns.build_payload_network_surrogate()

    def test_digest_document_binds_contract_support_residuals_and_sources(self) -> None:
        document = self.model.preflight_document()
        self.assertEqual(
            document["contract"]["latency_min_support"],
            ns.LATENCY_MIN_SUPPORT,
        )
        self.assertEqual(
            document["contract"]["udp_payload_capacity_bytes"], 12_492
        )
        self.assertEqual(
            document["source_binding"]["action_summary_sha256"],
            ACTION_SUMMARY_SHA256,
        )
        self.assertEqual(
            document["source_binding"]["surrogate_implementation_sha256"],
            self.model.surrogate_implementation_sha256,
        )
        # canonical_sha256 adds fitted curves, support knots, CV residuals,
        # semantic derivation/source bindings, executable implementation hash
        # and a digest of all 288 extracted observations before hashing.
        digest = self.model.canonical_sha256()
        self.assertEqual(len(digest), 64)

    def test_uncertainty_labels_make_no_block_bootstrap_claim(self) -> None:
        row = min(
            (
                row
                for row in self.model.observations
                if row.network_profile == "MID_VARIABLE"
                and row.network_support >= ns.LATENCY_MIN_SUPPORT
            ),
            key=lambda item: item.payload_bytes,
        )
        document = self.model.predict(
            network_profile=row.network_profile,
            payload_bytes=row.payload_bytes,
            datagram_count=row.datagram_count,
        ).to_canonical_dict()
        self.assertIn("BINOMIAL_REFERENCE_ONLY", json.dumps(document))
        self.assertIn("BLOCK_BOOTSTRAP_NOT_IMPLEMENTED", json.dumps(document))
        self.assertEqual(
            document["latency"]["condition"], ns.LATENCY_EVENT_DEFINITION
        )
        self.assertNotIn("EDGE_ADMISSION_WITH_NETWORK_TIMING", json.dumps(document))


class ReadOnlyCliTest(PayloadNetworkSurrogateTestBase):
    def test_preflight_cli_emits_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(ns.main(["--preflight"]), 0)
        document = json.loads(output.getvalue())
        self.assertEqual(document["status"], "PREFLIGHT_COMPLETE")
        self.assertEqual(document["inventory"]["cells"], 288)

    def test_report_cli_is_explicitly_offline_modeled(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(ns.main(["--report"]), 0)
        report = output.getvalue()
        self.assertIn(ns.EVIDENCE_CLASS, report)
        self.assertIn("not a measured or counterfactual per-frame causal", report)
        self.assertIn("not causal pre-admission arrival latency", report)
        self.assertIn("Held-entire-mode-out", report)

    def test_predict_cli_emits_one_prediction(self) -> None:
        row = min(
            (
                row
                for row in self.model.observations
                if row.network_profile == "FAVORABLE_STABLE"
            ),
            key=lambda row: row.payload_bytes,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                ns.main(
                    [
                        "--predict",
                        "--network-profile",
                        row.network_profile,
                        "--payload-bytes",
                        str(row.payload_bytes),
                        "--datagram-count",
                        str(row.datagram_count),
                    ]
                ),
                0,
            )
        document = json.loads(output.getvalue())
        self.assertEqual(document["evidence_class"], ns.EVIDENCE_CLASS)
        self.assertEqual(document["network_profile"], "FAVORABLE_STABLE")


if __name__ == "__main__":
    unittest.main()

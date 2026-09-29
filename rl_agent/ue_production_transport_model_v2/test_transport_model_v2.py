#!/usr/bin/env python3
"""Phase E tests: artifact parity, refusal, and preservation.

No test launches a radio, a container, CARLA or a live capture.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import model_v2 as M


def _synthetic_rows(n: int = 600) -> list[dict]:
    """Deterministic rows with a genuine monotone structure."""
    rng = np.random.default_rng(11)
    rows = []
    for index in range(n):
        backlog = float(rng.choice([0, 0, 0, 500, 5_000, 250_000, 2_000_000]))
        wire = float(rng.choice([6_500, 375_000, 620_000]))
        mcs = int(rng.integers(8, 29))
        latency = (40.0 + 9e-5 * wire + 5e-5 * backlog - 20.0 * (mcs / 28.0)
                   + float(rng.normal(0, 3)))
        on_time = latency <= 170.0 and backlog < 1_000_000
        # A synthetic but conservation-consistent successor state, so the
        # queue head has uncensored transitions to learn from.
        service = 400_000.0 + 40_000.0 * mcs
        successor = max(0.0, backlog + 2 * wire - service)
        rows.append({
            "cell_id": f"cell_{index % 6}", "partition": V1.FIT,
            "frame_index": index * 2,
            "pre_enqueue_backlog_bytes": int(backlog),
            "prior_ul_mcs": mcs, "action_wire_bytes": int(wire),
            "held_action_wire_bytes": int(wire),
            "deterministic_action_ingress_bytes": int(2 * wire),
            "transport_latency_ns": int(max(1.0, latency) * 1e6),
            "completed_within_deadline": bool(on_time),
            "terminal_outcome": ("COMPLETE_WITHIN_DEADLINE" if on_time
                                 else "COMPLETE_AFTER_DEADLINE"),
            "has_successor": True,
            "successor_backlog_bytes": int(successor),
        })
    return rows


class PreservationTests(unittest.TestCase):
    def test_v1_artifacts_are_preserved_byte_for_byte(self) -> None:
        self.assertTrue(C2.verify_preserved()["verified"])

    def test_v1_gate5_is_recorded_as_failed_and_never_rewritten(self) -> None:
        self.assertEqual(C2.V1_GATE5_STATUS,
                         "FAILED__PRESERVED__NEVER_REWRITTEN_AS_PASSED")
        for result in C2.PRESERVED_GATE5_RESULTS:
            self.assertFalse(result["passed"])
        self.assertFalse(C2.CONFIRMATORY)
        self.assertIn("NO_LONGER_PRISTINE", C2.VALIDATION_POPULATION_STATUS)


class BacklogNormalizationTests(unittest.TestCase):
    def test_reference_is_the_verified_oai_ceiling(self) -> None:
        self.assertEqual(C2.RLC_AM_TX_ADMISSION_CEILING_BYTES, 50_000_000)
        self.assertFalse(C2.BACKLOG_REFERENCE_IS_FIT_DERIVED)
        report = C2.verify_oai_ceiling()
        self.assertTrue(report["verified"])
        self.assertEqual(report["ceiling_bytes"], 50_000_000)
        self.assertEqual(
            C2.OAI_CEILING_DERIVATION["rlc_tx_maxsize_define"]
            * C2.OAI_CEILING_DERIVATION["am_multiplier"], 50_000_000)

    def test_registered_mappings(self) -> None:
        expected = {0: 0.000, 40_000: 0.598, 1_000_000: 0.779,
                    10_000_000: 0.909, 50_000_000: 1.000}
        for value, target in expected.items():
            self.assertAlmostEqual(C2.backlog_scaled(value), target, places=3)

    def test_zero_is_exactly_zero_and_clip_is_exactly_one(self) -> None:
        self.assertEqual(C2.backlog_scaled(0), 0.0)
        self.assertEqual(C2.backlog_scaled(50_000_000), 1.0)
        self.assertEqual(C2.backlog_scaled(900_000_000), 1.0)

    def test_legacy_scale_is_rejected(self) -> None:
        self.assertEqual(C2.REJECTED_LEGACY_BACKLOG_SCALE, 1.0)
        self.assertNotEqual(C2.RLC_AM_TX_ADMISSION_CEILING_BYTES,
                            C2.REJECTED_LEGACY_BACKLOG_SCALE)

    def test_never_described_as_percentage_full(self) -> None:
        self.assertIn("NEVER_PERCENTAGE_FULL", C2.BACKLOG_FEATURE_SEMANTICS)
        # 40 kB is 0.598 scaled but only 0.08% physically occupied
        row = next(r for r in C2.backlog_mapping_report()
                   if r["bytes"] == 40_000)
        self.assertAlmostEqual(row["log_scaled_backlog"], 0.598, places=3)
        self.assertLess(row["physical_occupancy_fraction"], 0.001)

    def test_no_capacity_or_drain_rate_feature(self) -> None:
        for name in ("queue_capacity_bytes", "drain_rate_bps"):
            self.assertIn(name, C2.FORBIDDEN_ADDITIONAL_QUEUE_FEATURES)


class CausalContractTests(unittest.TestCase):
    def test_cutoff_is_frame_open_not_first_send(self) -> None:
        self.assertEqual(C2.CAUSAL_CUTOFF_FIELD, "frame_open_monotonic_ns")
        self.assertEqual(C2.CAUSAL_CUTOFF_REJECTED_FIELD,
                         "first_send_monotonic_ns")

    def test_forbidden_predictors_are_excluded(self) -> None:
        for name in ("harq_round", "retransmission_count", "datagram_count",
                     "profile_id", "target_snr_db",
                     "measured_rlc_ingress_bytes", "transport_latency_ns"):
            self.assertIn(name, C2.FORBIDDEN_PREDICTORS)
            self.assertNotIn(name, C2.ALLOWED_PREDICTORS)
        self.assertEqual(
            set(C2.ALLOWED_PREDICTORS) & set(C2.FORBIDDEN_PREDICTORS), set())

    def test_separate_monotone_effects_required(self) -> None:
        self.assertTrue(C2.SEPARATE_MONOTONE_EFFECTS)
        self.assertTrue(C2.SHARED_BACKLOG_PLUS_BYTES_COEFFICIENT_FORBIDDEN)

    def test_reward_error_correspondence(self) -> None:
        self.assertAlmostEqual(
            C2.reward_error_for_latency_error_ms(17.0), 0.025, places=4)
        self.assertAlmostEqual(
            C2.reward_error_for_latency_error_ms(34.0), 0.050, places=4)


class ModelShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _synthetic_rows()
        cls.model = M.fit_model(cls.rows)

    def test_coefficients_are_sign_constrained(self) -> None:
        d = self.model.deadline
        for value in (d.w_backlog, d.w_bytes, d.w_mcs):
            self.assertGreaterEqual(value, 0.0)
        l = self.model.latency
        for value in (l.a_backlog, l.b_bytes, l.c_mcs):
            self.assertGreaterEqual(value, 0.0)

    def test_backlog_and_bytes_have_separate_coefficients(self) -> None:
        l = self.model.latency
        self.assertNotEqual(l.a_backlog, l.b_bytes)

    def test_latency_is_generatively_bounded_not_clipped(self) -> None:
        """Every reachable prediction lies strictly inside (0, 170]."""
        extremes = M.Features(
            backlog_scaled=np.array([0.0, 1.0, 0.5, 1.0, 0.0]),
            backlog_bytes=np.array([0.0, 5e7, 1e6, 5e7, 0.0]),
            bytes_mb=np.array([0.0, 100.0, 0.4, 0.0, 100.0]),
            wire_bytes=np.array([0.0, 1e8, 4e5, 0.0, 1e8]),
            mcs_norm=np.array([1.0, 0.0, 0.5, 1.0, 0.0]))
        values = self.model.latency.latency_ms(extremes)
        self.assertTrue(bool(np.all(values > 0.0)))
        self.assertTrue(bool(np.all(values <= 170.0)))

    def test_queue_head_is_causal_and_monotone_in_mcs(self) -> None:
        queue = self.model.queue
        bins = sorted(queue.service_by_mcs_bin)
        values = [queue.service_by_mcs_bin[b] for b in bins]
        self.assertEqual(values, sorted(values),
                         "service must be non-decreasing in the MCS bin")
        # conservation form and the zero clamp
        self.assertEqual(
            queue.next_backlog_bytes(backlog_bytes=0.0, ingress_bytes=0.0,
                                     mcs=20)[0], 0.0)
        grew = queue.next_backlog_bytes(
            backlog_bytes=5_000_000.0, ingress_bytes=1_200_000.0, mcs=8)[0]
        self.assertGreater(grew, 0.0)

    def test_monotone_in_every_axis(self) -> None:
        self.assertEqual(
            M.monotonicity_violations(self.model)["violations"], 0)

    def test_deterministic_under_the_frozen_seed(self) -> None:
        again = M.fit_model(self.rows)
        self.assertEqual(again.deadline.to_dict(),
                         self.model.deadline.to_dict())
        self.assertEqual(again.latency.to_dict(),
                         self.model.latency.to_dict())


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _synthetic_rows()
        cls.model = M.fit_model(cls.rows)
        cls.metrics = {
            "brier": 0.02, "false_success_rate": 0.01,
            "latency_abs_error_p50_ms": 8.0,
            "latency_abs_error_p95_ms": 30.0,
            "reward_error_p50": 0.012, "reward_error_p95": 0.045,
            "n": len(cls.rows), "n_on_time": 400, "confusion": {},
        }
        cls.document = A2.export_artifact(
            model=cls.model, cv_metrics=cls.metrics, causal_coverage=1.0,
            feature_provenance_passed=True, monotonicity_violations=0,
            action_ranking={"model_agrees_with_empirical_ranking_fraction": 1.0},
            raw_backlog_support={"fit_min_bytes": 0.0,
                                 "fit_max_bytes": 2_000_000.0,
                                 "min_bytes": 0.0, "max_bytes": 2_000_000.0,
                                 "above_oai_ceiling": 0})

    def _write(self, document) -> Path:
        path = Path(tempfile.mkdtemp()) / "artifact.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_export_load_predict_round_trip_is_exact(self) -> None:
        loaded = A2.ProductionTransportModelV2.load(self._write(self.document))
        for backlog, wire, mcs in ((0, 6_500, 20), (5_000, 375_000, 12),
                                   (1_500_000, 620_000, 28)):
            prediction = loaded.predict(pre_enqueue_backlog_bytes=backlog,
                                        wire_bytes=wire, prior_ul_mcs=mcs)
            features = M.Features(
                backlog_scaled=np.array([C2.backlog_scaled(float(backlog))]),
                backlog_bytes=np.array([float(backlog)]),
                bytes_mb=np.array([wire / M.BYTES_SCALE]),
                wire_bytes=np.array([float(wire)]),
                mcs_norm=np.array([mcs / M.MCS_SCALE]))
            self.assertEqual(prediction.on_time_probability,
                             float(self.model.deadline.probability(features)[0]))
            self.assertEqual(prediction.conditional_latency_ms,
                             float(self.model.latency.latency_ms(features)[0]))

    def test_out_of_support_is_refused_not_extrapolated(self) -> None:
        loaded = A2.ProductionTransportModelV2.load(self._write(self.document))
        with self.assertRaises(A2.OutOfSupport):
            loaded.predict(pre_enqueue_backlog_bytes=40_000_000,
                           wire_bytes=6_500, prior_ul_mcs=20)
        with self.assertRaises(A2.OutOfSupport):
            loaded.predict(pre_enqueue_backlog_bytes=0, wire_bytes=5_000,
                           prior_ul_mcs=20)
        with self.assertRaises(A2.OutOfSupport):
            loaded.predict(pre_enqueue_backlog_bytes=0, wire_bytes=6_500,
                           prior_ul_mcs=2)

    def test_normalization_clip_cannot_hide_out_of_support_backlog(self) -> None:
        """Both values clip to 1.0 yet both must still be refused."""
        loaded = A2.ProductionTransportModelV2.load(self._write(self.document))
        self.assertEqual(C2.backlog_scaled(50_000_000), 1.0)
        self.assertEqual(C2.backlog_scaled(80_000_000), 1.0)
        for value in (50_000_000, 80_000_000):
            with self.assertRaises(A2.OutOfSupport):
                loaded.predict(pre_enqueue_backlog_bytes=value,
                               wire_bytes=6_500, prior_ul_mcs=20)

    def test_failed_gate_artifact_is_refused(self) -> None:
        broken = copy.deepcopy(self.document)
        broken["metrics"]["latency_abs_error_p95_ms"] = 99.0
        broken["gate_verdicts"] = A2.gate_verdicts(broken["metrics"])
        broken["all_gates_passed"] = all(broken["gate_verdicts"].values())
        with self.assertRaises(A2.ArtifactError):
            A2.ProductionTransportModelV2.load(self._write(broken))

    def test_flipping_all_gates_passed_alone_does_not_enable_it(self) -> None:
        broken = copy.deepcopy(self.document)
        broken["metrics"]["brier"] = 0.9
        broken["all_gates_passed"] = True          # the lie
        with self.assertRaises(A2.ArtifactError):
            A2.ProductionTransportModelV2.load(self._write(broken))
        # and also when the verdict map is forged wholesale
        forged = copy.deepcopy(self.document)
        forged["metrics"]["brier"] = 0.9
        forged["gate_verdicts"] = {k: True for k in forged["gate_verdicts"]}
        forged["all_gates_passed"] = True
        with self.assertRaises(A2.ArtifactError):
            A2.ProductionTransportModelV2.load(self._write(forged))

    def test_foreign_or_unbound_artifact_is_refused(self) -> None:
        for key, value in (("schema", "other.v1"), ("version", 1),
                           ("contract_v2_sha256", "0" * 64),
                           ("v1_contract_sha256", "0" * 64),
                           ("adding_both_components_is_forbidden", False)):
            broken = copy.deepcopy(self.document)
            broken[key] = value
            with self.assertRaises(A2.ArtifactError):
                A2.ProductionTransportModelV2.load(self._write(broken))

    def test_288_component_is_replaced_never_added(self) -> None:
        self.assertEqual(self.document["replaces_288_component"],
                         V1.REPLACED_288_COMPONENT)
        self.assertTrue(
            self.document["adding_both_components_is_forbidden"])
        self.assertEqual(self.document["shared_endpoint"],
                         V1.SHARED_ENDPOINT)

    def test_artifact_declares_its_evidence_class(self) -> None:
        self.assertEqual(self.document["evidence_class"],
                         "POSTHOC_DEVELOPMENT_MODEL_FOR_EXPLORATORY_RUN4_TRAINING")
        self.assertFalse(self.document["confirmatory"])
        self.assertIn("V1_GATE5_REMAINS_FAILED", self.document["disclosures"])
        self.assertIn("NOT_P_ADMIT__INTERNAL_TRAINING_ENVIRONMENT_MODEL_ONLY",
                      self.document["disclosures"])

    def test_next_backlog_uses_deterministic_ingress_only(self) -> None:
        loaded = A2.ProductionTransportModelV2.load(self._write(self.document))
        value = loaded.predict_next_backlog_bytes(
            pre_enqueue_backlog_bytes=100_000,
            deterministic_action_ingress_bytes=13_000, prior_ul_mcs=20)
        self.assertGreaterEqual(value, 0.0)
        self.assertEqual(
            loaded.predict_next_backlog_bytes(
                pre_enqueue_backlog_bytes=0,
                deterministic_action_ingress_bytes=0, prior_ul_mcs=20), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

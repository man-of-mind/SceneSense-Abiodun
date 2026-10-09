"""Collector-level parity: outcomes/state, session reset, holds, quality/scaling unchanged, determinism, resume.

Loads the FIT grid once (~2 s CPU). Uses a test-only relaxed binding (mode-level strata, wide scene support,
FAVORABLE profile label) so mechanics can run; the strict fixture binding is shown to block immediately.
"""
import copy
import json
import sqlite3
import unittest
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import calibration as K
from . import chains as CH
from . import collector as COL
from . import outcome as O

BINDING = K.PACKAGE_DIR / "bindings" / "LIVE100_REGRESSION_BINDING_V2.json"
SOURCES = None
ACTIONS = [(8, 3500), (5, 6700), (8, 4200), (10, 3000), (5, 6000)]


def setUpModule():
    global SOURCES
    SOURCES = COL.load_shared_sources()


def _relaxed_calibration():
    doc = json.loads(BINDING.read_text())
    doc["policy"]["action_match"]["q_tolerance_e4"] = 9800
    doc["policy"]["scene_conditioning"]["support_margin_fraction"] = 50.0
    doc["policy"]["uplink"] = {"mode": "CHAIN_ONLY", "transport_artifact_sha256": K.TRANSPORT_ARTIFACT_SHA256,
                               "chain_only_mcs_tolerance": 28}
    return K.CalibrationSetV1(doc)


class FavorableRun4(COL.CorrectedRun4CollectorV1):
    """Test double: label the unlabeled Run-4 channel FAVORABLE so mechanics run against the fixture chains."""

    def _channel_profile(self):
        return "FAVORABLE_STABLE"


def _run(seed=3, n=160, cal=None):
    collector = FavorableRun4(calibration=cal or _relaxed_calibration(), seed=seed, shared_sources=SOURCES)
    for i in range(n):
        collector.collect(*ACTIONS[i % len(ACTIONS)])
    return collector


class StrictBindingBlocks(unittest.TestCase):
    def test_unsupported_terms_block_training(self):
        strict = K.CalibrationSetV1(json.loads(BINDING.read_text()))
        collector = COL.CorrectedRun4CollectorV1(calibration=strict, seed=3, shared_sources=SOURCES)
        with self.assertRaises(CH.UncalibratedTermError):
            collector.collect(8, 3500)        # unmeasured exact action and unlabeled channel


class OutcomeAndStateParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = _run()
        cls.records = cls.collector.records()

    def test_all_live_outcome_classes_occur_and_follow_the_controller(self):
        terminals = {r["terminal"] for r in self.records}
        self.assertEqual(terminals, {"SUCCESS", "TIMEOUT", "EVALUATOR_FAULT"})
        for r in self.records:
            expected = O.classify(O.receipt_elapsed_ns(r["receipt_ms"]), r["q_perc_grid"])
            self.assertEqual(r["event_kind"], expected.kind.value)
            if r["terminal"] == "SUCCESS":
                self.assertLessEqual(r["resolution_elapsed_ns"], O.DEADLINE_NS)
                self.assertAlmostEqual(r["reward"], r["q_perc_grid"] - 0.25 * r["latency_ms"] / 170.0, places=12)
            elif r["terminal"] == "TIMEOUT":
                self.assertEqual(r["resolution_elapsed_ns"], O.DEADLINE_NS + 1)
                self.assertEqual(r["reward"], -1.0)
                self.assertTrue(r["receipt_ms"] is None or O.receipt_elapsed_ns(r["receipt_ms"]) > O.DEADLINE_NS)
            else:
                self.assertFalse(r["learning_included"])
                self.assertIsNone(r["q_perc_grid"])
                self.assertLessEqual(r["resolution_elapsed_ns"], O.DEADLINE_NS)
        late_undefined = [r for r in self.records if r["q_perc_grid"] is None and r["terminal"] == "TIMEOUT"]
        self.assertTrue(late_undefined, "late undefined quality must appear as a plain timeout")

    def test_next_state_encodes_exactly_the_previous_outcome(self):
        for previous, current in zip(self.records, self.records[1:]):
            features = [float.fromhex(x) for x in current["state_features"]]
            if previous["terminal"] == "EVALUATOR_FAULT":
                self.assertNotEqual(current["session_uuid"], previous["session_uuid"])
                self.assertEqual(current["decision_seq"], 0)
                self.assertFalse(current["previous_present"])
                kind = None
            else:
                self.assertEqual(current["session_uuid"], previous["session_uuid"])
                self.assertEqual(current["decision_seq"], previous["decision_seq"] + 1)
                kind = (contract.RewardEventKind.DELIVERED_SUCCESS if previous["terminal"] == "SUCCESS"
                        else contract.RewardEventKind.TIMEOUT)
            expected = O.previous_outcome_features(kind, mode_id=previous["mode_id"], q_e4=previous["q_e4"],
                                                   q_perc=previous["q_perc_grid"], latency_ms=previous["latency_ms"])
            self.assertEqual([f.hex() for f in features[4:21]], [float(x).hex() for x in expected])

    def test_holds_are_two_tensors_with_one_reward_request(self):
        for r in self.records:
            if r["learning_included"]:
                self.assertEqual(r["reward_request_flags"], [True, False])
            self.assertEqual(r["ingress_bytes"], r["wire_reward"] + r["wire_held"])

    def test_receipt_comes_from_a_measured_chain_composition(self):
        for r in self.records:
            vector = self.collector.calibration.vectors[r["chain_id"]]
            composed = CH.compose(vector, carry_in=r["score_channel_carry_in_ms"])   # CHAIN_ONLY: no uplink shift
            self.assertAlmostEqual(r["receipt_ms"], composed.receipt, places=9)
            self.assertAlmostEqual(sum(v for _, v in r["critical_path"]), r["receipt_ms"], places=9)

    def test_score_channel_occupancy_carries_into_the_next_decision(self):
        self.assertIsNone(self.records[0]["score_channel_carry_in_ms"])
        for previous, current in zip(self.records, self.records[1:]):
            busy_ns = previous["opened_ns"] + int(round(previous["score_channel_busy_until_ms"] * 1_000_000))
            self.assertAlmostEqual(current["score_channel_carry_in_ms"], (busy_ns - current["opened_ns"]) / 1e6,
                                   places=9)


class QualityAndScalingUnchanged(unittest.TestCase):
    def test_defined_scene_quality_equals_the_original_catalog(self):
        connect = sqlite3.connect

        def immutable(database, *args, **kwargs):     # avoid touching the registered DB's -shm timestamp
            return connect(database + "&immutable=1" if database.startswith("file:") else database, *args, **kwargs)
        with mock.patch.object(SS.sqlite3, "connect", immutable):
            original = SS.FitSceneCatalog()
        catalog = SOURCES["catalog"]
        self.assertEqual(set(original.keys), set(catalog.defined_keys))
        self.assertEqual(len(catalog.partially_undefined_keys), 36)
        for key in original.keys[::37]:
            for mode_id, q_e4 in ((0, 0), (5, 6700), (8, 3500), (11, 9800), (7, 4321)):
                ref = original.draw(key, mode_id=mode_id, q_e4=q_e4)
                q, total, wire, anchor, _, _ = catalog.draw(key, mode_id=mode_id, q_e4=q_e4)
                self.assertEqual((q, total, wire, anchor),
                                 (ref.q_perc, ref.total_transmitted_bytes, ref.wire_bytes, ref.is_registered_anchor))

    def test_state_scaling_is_the_training_scaling(self):
        collector = _run(n=2)
        self.assertIs(collector._provider.scaling, live_state_v2.TRAINING_SCALING)


class DeterminismAndResume(unittest.TestCase):
    def test_same_seed_same_records_and_seeds_differ(self):
        a, b, c = _run(seed=11, n=60), _run(seed=11, n=60), _run(seed=12, n=60)
        self.assertEqual(a.records(), b.records())
        self.assertNotEqual(a.records(), c.records())

    def test_checkpoint_restore_continues_identically(self):
        uninterrupted = _run(seed=22, n=120)
        # Seed 22 has a timely undefined-quality session break before the checkpoint, so resume crosses it.
        self.assertTrue(any(r["terminal"] == "EVALUATOR_FAULT" for r in uninterrupted.records()[:70]))
        first = _run(seed=22, n=70)
        checkpoint = json.loads(json.dumps(first.checkpoint()))
        self.assertGreaterEqual(checkpoint["session_index"], 1)
        resumed = FavorableRun4(calibration=_relaxed_calibration(), seed=22, shared_sources=SOURCES)
        resumed.restore(checkpoint)
        for i in range(70, 120):
            resumed.collect(*ACTIONS[i % len(ACTIONS)])
        self.assertEqual(resumed.records(), uninterrupted.records())

    def test_restore_rejects_foreign_checkpoints(self):
        collector = _run(seed=21, n=5)
        checkpoint = collector.checkpoint()
        other = FavorableRun4(calibration=_relaxed_calibration(), seed=22, shared_sources=SOURCES)
        with self.assertRaises(COL.CollectorError):
            other.restore(checkpoint)
        tampered = dict(checkpoint, records_sha256="0" * 64)
        with self.assertRaises(COL.CollectorError):
            FavorableRun4(calibration=_relaxed_calibration(), seed=21, shared_sources=SOURCES).restore(tampered)
        doc = json.loads(BINDING.read_text())
        doc["policy"]["action_match"]["q_tolerance_e4"] = 9700
        doc["policy"]["scene_conditioning"]["support_margin_fraction"] = 50.0
        with self.assertRaises(COL.CollectorError):
            FavorableRun4(calibration=K.CalibrationSetV1(doc), seed=21, shared_sources=SOURCES).restore(checkpoint)


class Run5Variant(unittest.TestCase):
    def test_22d_state_profile_label_and_uplink_transfer(self):
        doc = json.loads(BINDING.read_text())
        doc["policy"]["action_match"]["q_tolerance_e4"] = 9800
        doc["policy"]["scene_conditioning"]["support_margin_fraction"] = 50.0
        doc["policy"]["uplink"]["validation"]["scope"].update(mcs_range=[9, 28], wire_range=[73165, 406912])
        collector = COL.CorrectedRun5CollectorV1(calibration=K.CalibrationSetV1(doc), seed=0, shared_sources=SOURCES)
        for _ in range(40):                       # seed 0 starts in a FAVORABLE_STABLE segment
            collector.collect(8, 3500)
        records = collector.records()
        self.assertTrue(all(len(r["state_features"]) == 22 and r["profile"] == "FAVORABLE_STABLE" for r in records))
        self.assertTrue(any(abs(r["uplink_delta_ms"]) > 0 for r in records))

    def test_unmeasured_profile_blocks(self):
        strict = K.CalibrationSetV1(json.loads(BINDING.read_text()))
        collector = COL.CorrectedRun5CollectorV1(calibration=strict, seed=1, shared_sources=SOURCES)
        with self.assertRaises(CH.UncalibratedTermError):
            for _ in range(400):
                collector.collect(8, 3500)


if __name__ == "__main__":
    unittest.main()

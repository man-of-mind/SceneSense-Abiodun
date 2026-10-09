"""Evaluator tests on a synthetic FIT-scene fixture (no held-scene outcome is produced)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_collector as RC
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_evaluator as EV
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_held_scene_partition as H
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR

WORKTREE = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(os.environ.get("RUN5_EVIDENCE_ROOT", WORKTREE.parent / "abiodun")).resolve()
FIXTURE_DECISIONS = 12


class FitFixtureCatalog:
    """Synthetic stand-in for the held catalogue built from FIT scenes only."""

    def __init__(self, catalog, count: int) -> None:
        self._catalog = catalog
        self._keys = tuple(catalog.keys[:count])
        self.binding_sha256 = "f" * 64

    keys = property(lambda self: self._keys)
    scene_count = property(lambda self: len(self._keys))

    def scene_descriptors(self, key):
        return self._catalog.scene_descriptors(key)

    def draw(self, key, **kwargs):
        return self._catalog.draw(key, **kwargs)


class EvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.shared = RC.build_shared_sources(EV.ARTIFACT, EVIDENCE_ROOT)
        cls.fixture = FitFixtureCatalog(cls.shared["catalog"], FIXTURE_DECISIONS)
        cls.evidence = json.loads((EV.PACKAGE / "RUN5_DEEP_CAMPAIGN_EVIDENCE.json").read_text())
        cls.specs = EV.policy_specs(cls.evidence, {"mode_id": 11, "q_e4": 9800})
        cls.catalogue = EV.anchors()

    def collector(self, profile="MID_VARIABLE", seed=9017):
        return EV.Run5EvaluationCollectorV1(shared_sources=self.shared, held_catalog=self.fixture,
                                            profile=profile, validation_seed=seed,
                                            evidence_root=EVIDENCE_ROOT)

    def trajectory(self, spec, profile="MID_VARIABLE", seed=9017, act=None):
        tape = EV.snr_tape(self.shared, profile, seed, FIXTURE_DECISIONS)
        shuffled = [tape[i] for i in EV.shuffle_permutation(profile, seed, FIXTURE_DECISIONS)]
        policy, actor = EV.make_policy(spec, EVIDENCE_ROOT, shuffled)
        collector = self.collector(profile, seed)
        rows, exogenous = EV.run_trajectory(
            spec=spec, act=act or policy, collector=collector, decisions=FIXTURE_DECISIONS,
            catalogue=self.catalogue, reference_actor=actor if spec.family == "RUN5" else None,
            shuffled_values=shuffled)
        return rows, exogenous, collector, tape, shuffled

    def spec(self, policy_id):
        return next(s for s in self.specs if s.policy_id == policy_id)

    # -- registered design -------------------------------------------------
    def test_policy_set_row_counts_and_single_registered_final(self) -> None:
        self.assertEqual(len(self.specs), 38)
        finals = [s.policy_id for s in self.specs if s.registered_final and s.family == "RUN5"]
        self.assertEqual(finals, [f"run5_seed{s}_u10000" for s in (17, 29, 43)])
        self.assertEqual(len(EV.contexts()), 12)
        self.assertEqual({p for p, _ in EV.contexts()}, set(EV.PROFILES))
        self.assertEqual({s for _, s in EV.contexts()}, {9017, 9029, 9043})
        self.assertEqual(12 * 38 * 255, 116_280)
        self.assertEqual(len(self.catalogue), 72)

    def test_held_partition_eligibility_rule_and_disjointness(self) -> None:
        full = {(m, q): (0.5, 1, 1, "x") for m in range(12) for q in H.SS.Q_E4_GRID}
        self.assertEqual(H.eligibility({"camera_si": 1.0, "radar_p40": 0.2, "grid": full}), [])
        for entry, reason in (
                ({"camera_si": None, "radar_p40": 0.2, "grid": full}, "CAMERA_SI_MISSING"),
                ({"camera_si": 1.0, "radar_p40": None, "grid": full}, "RADAR_P40_MISSING"),
                ({"camera_si": 1.0, "radar_p40": 0.2, "grid": dict(list(full.items())[:5])},
                 "QUALITY_GRID_INCOMPLETE"),
                ({"camera_si": 1.0, "radar_p40": 0.2,
                  "grid": {**full, (0, 0): (None, 1, 1, "x")}}, "UNDEFINED_Q_PERC_ROWS_1")):
            self.assertIn(reason, H.eligibility(entry))
        sealed = json.loads(H.SEALED_PARTITION.read_text())
        self.assertEqual((sealed["candidate_count"], sealed["eligible_count"]), (256, 255))
        self.assertFalse(set(sealed["eligible_keys"]) & set(self.shared["catalog"].keys))
        self.assertEqual(sealed["eligible_keys"],
                         sorted(sealed["eligible_keys"], key=H._order))

    # -- tapes and per-policy history ------------------------------------------
    def test_identical_exogenous_tape_and_separate_histories(self) -> None:
        a = self.trajectory(EV.PolicySpec("fixed_a", "FIXED_ACTION", mode_id=11, q_e4=9800))
        b = self.trajectory(EV.PolicySpec("fixed_b", "FIXED_ACTION", mode_id=11, q_e4=5000))
        c = self.trajectory(self.spec("run5_seed17_u10000"))
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[1], c[1])
        self.assertEqual([r["scene"] for r in a[0]], list(self.fixture.keys))
        self.assertNotEqual([r["reward"] for r in a[0]], [r["reward"] for r in b[0]])
        self.assertIsNot(a[2], b[2])                       # separate collectors/backlogs
        prev_a = [r.state[4:21] for r in a[2].history()[1:]]
        prev_b = [r.state[4:21] for r in b[2].history()[1:]]
        self.assertNotEqual(prev_a, prev_b)
        for rows in (a[0], b[0], c[0]):
            self.assertEqual(len(rows), FIXTURE_DECISIONS)
            self.assertFalse(any(r["fault"] for r in rows))

    def test_reset_only_at_trajectory_start(self) -> None:
        rows, _, collector, _, _ = self.trajectory(self.spec("run4_seed43_u10000"))
        history = collector.history()
        self.assertEqual([r.decision_seq for r in history], list(range(FIXTURE_DECISIONS)))
        self.assertEqual(history[0].state[19], 0.0)
        self.assertTrue(all(r.state[19] == 1.0 for r in history[1:]))
        self.assertEqual(len({r.session_uuid for r in history}), 1)

    def test_kernel_mirror_oracle_and_tape_consistency(self) -> None:
        rows, exogenous, _, tape, shuffled = self.trajectory(
            EV.PolicySpec("fixed_a", "FIXED_ACTION", mode_id=11, q_e4=9800))
        for row in rows:
            self.assertGreaterEqual(row["oracle_expected_reward"], row["expected_reward"])
            self.assertTrue(-1.0 <= row["expected_reward"] <= 1.0)
        self.assertEqual([float.fromhex(t["snr_db"]) for t in exogenous], tape)
        self.assertEqual(sorted(shuffled), sorted(tape))
        self.assertTrue(all(5.5 <= v <= 24.5 for v in shuffled))
        self.assertEqual(EV.shuffle_permutation("MID_VARIABLE", 9017, 12),
                         EV.shuffle_permutation("MID_VARIABLE", 9017, 12))
        self.assertNotEqual(EV.shuffle_permutation("MID_VARIABLE", 9017, 50),
                            EV.shuffle_permutation("ADVERSE_STABLE", 9017, 50))

    def test_shuffled_snr_changes_only_the_actor_input(self) -> None:
        true = self.trajectory(self.spec("run5_seed17_u10000"))
        shuf = self.trajectory(self.spec("run5_shuffled_snr_seed17_u10000"))
        self.assertEqual(true[1], shuf[1])                 # same exogenous channel
        self.assertEqual([r["snr_db"] for r in true[0]], [r["snr_db"] for r in shuf[0]])
        self.assertTrue(all(r["shuffled_probe"] is not None for r in true[0]))
        self.assertTrue(all(r["shuffled_probe"] is None for r in shuf[0]))
        for record, diag in zip(shuf[2].history(), shuf[2].diagnostics()):
            # the recorded state keeps the TRUE causal SNR; only the actor input is shuffled
            self.assertEqual(record.state[21], SNR.scale_snr_db(diag["snr_db"]))

    def test_run4_comparator_sees_features_0_to_20_only(self) -> None:
        seen = []
        policy, _ = EV.make_policy(self.spec("run4_seed43_u10000"), EVIDENCE_ROOT, None)

        def spy(k, state):
            seen.append(len(state))
            return policy(k, state)

        rows = self.trajectory(self.spec("run4_seed43_u10000"), act=spy)[0]
        self.assertEqual(set(seen), {22})           # the wrapper slices to 21 internally
        self.assertEqual(len(rows), FIXTURE_DECISIONS)

    def test_run5_actor_hash_binding_is_enforced(self) -> None:
        spec = self.spec("run5_seed29_u01500")
        EV.load_run5_actor(spec)
        with self.assertRaises(EV.EvaluatorError):
            EV.load_run5_actor(EV.PolicySpec(**{**spec.__dict__, "actor_tree_sha256": "0" * 64}))

    # -- faults and unconditional metrics ---------------------------------------
    def test_faults_are_recorded_not_dropped(self) -> None:
        def broken(k, state):
            if k == 3:
                raise RuntimeError("synthetic evaluator fault")
            return 11, 9800
        rows = self.trajectory(EV.PolicySpec("fault", "FIXED_ACTION", mode_id=11, q_e4=9800),
                               act=broken)[0]
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[-1]["fault"])
        metrics = EV.trajectory_metrics(rows)
        self.assertEqual((metrics["decisions"], metrics["faults"]), (3, 1))

    def test_unconditional_metrics_count_timeouts(self) -> None:
        base = {"fault": False, "mode_id": 0, "q_e4": 9800, "executed_q_perc": 0.5,
                "expected_reward": 0.1, "oracle_expected_reward": 0.3,
                "oracle_realized_reward": 0.4, "oracle_refused_anchors": 2}
        rows = [{**base, "terminal": "SUCCESS", "reward": 0.4, "latency_ms": 100.0,
                 "delivered_q_perc": 0.5},
                {**base, "terminal": "TIMEOUT", "reward": -1.0, "latency_ms": None,
                 "delivered_q_perc": None}]
        metrics = EV.trajectory_metrics(rows)
        self.assertAlmostEqual(metrics["unconditional_reward_mean"], -0.3)
        self.assertEqual((metrics["success_rate"], metrics["timeout_rate"]), (0.5, 0.5))
        self.assertEqual(metrics["latency_censored_count"], 1)
        self.assertAlmostEqual(metrics["oracle_realized_regret_mean"], (0.0 + 1.4) / 2)

    def test_sweep_refuses_without_a_matching_sealed_manifest(self) -> None:
        if not EV.MANIFEST_PATH.exists():
            with self.assertRaises(FileNotFoundError):
                EV.run_sweep(EVIDENCE_ROOT, Path(tempfile.mkdtemp()) / "out", 1)
        else:
            sealed = json.loads(EV.MANIFEST_PATH.read_text())
            self.assertEqual(sealed["row_counts"]["decision_rows_total"], 116_280)


if __name__ == "__main__":
    unittest.main()

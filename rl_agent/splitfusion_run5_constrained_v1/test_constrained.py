"""Focused mathematical / accounting checks for the constrained candidate (CPU). Checkpoint integrity is proven by the
campaign's smoke/resume test at update 250 (dual state included in the digest).

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m unittest -v \
    rl_agent.splitfusion_run5_constrained_v1.test_constrained
"""
from __future__ import annotations

import functools
import random
import unittest

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_reward_trials_v1 import catalog as K

from . import campaign as CAMP
from . import objective as O
from . import registration as REG
from . import trainer as TR


@functools.lru_cache(maxsize=None)
def doc():
    return REG.document()


@functools.lru_cache(maxsize=None)
def orchestrator(updates=110):
    torch.set_num_threads(4)
    d = doc()
    collector, _ = CAMP.factories(d, 29)
    o = TR.ConstrainedOrchestratorV1(collector_factory=collector, seed=29, authorization_sha256="t",
                                     lambda0=np.asarray(d["lambda0"]), objective_sha256=d["objective_sha256"])
    o.warmup_phase()
    for _ in range(updates):
        o.step_update()
    return o


class ConstrainedObjectiveTest(unittest.TestCase):
    def test_primary_is_registered_Y_without_segmentation_and_same_definedness(self):
        cat = O.LocCatalogV1("fit", "0" * 64)
        ref = K.TrialSceneCatalogV1("fit", "C1")
        rng = random.Random(7)
        for _ in range(1500):
            k, m = rng.choice(cat.keys), rng.randrange(12)
            q = rng.randint(*S.BOUNDS[m])
            y = cat.reward_value(k, mode_id=m, q_e4=q)
            c0, c1 = ref.value_for("C0", k, mode_id=m, q_e4=q), ref.value_for("C1", k, mode_id=m, q_e4=q)
            self.assertEqual(y is None, c0 is None)
            if y is not None:
                self.assertAlmostEqual(y, (3 * c1 - c0) / 2, places=12)     # C0 = Y(0.7+0.3S), C1 = Y(0.9+0.1S)

    def test_normalised_loc_constraint_equals_retention_minus_target(self):
        d = doc()
        cat = O.LocCatalogV1("fit", d["objective_sha256"])
        consts = O.fit_constants(cat)
        g = np.mean([O.g_vector(cat, k, 11, 7000, d["targets"], consts) for k in cat.keys], axis=0)
        for i, name in enumerate(O.CONSTRAINTS):
            if name.startswith("loc"):
                c = name.split("_")[1]
                hits = sum(cat.expected(k, mode_id=11, q_e4=7000)[c]["hits"] for k in cat.keys)
                ref = sum(cat.reference_components(k)[c]["hits"] for k in cat.keys)
                self.assertAlmostEqual(g[i], hits / ref - d["targets"][name], places=9)

    def test_replay_reward_is_record_reward_plus_lambda_dot_g_with_identical_sampling(self):
        o = orchestrator()
        gen_a, gen_b = torch.Generator(), torch.Generator()
        gen_a.manual_seed(5); gen_b.manual_seed(5)
        batch = o.replay.sample(64, gen_a)
        perm = torch.randperm(len(o.replay._rows), generator=gen_b)[:64]
        rows = [o.replay._rows[int(i)] for i in perm]
        plain = RT.Run5ReplayBufferV1.sample(o.replay, 64, (lambda g: (g.manual_seed(5), g)[1])(torch.Generator()))
        self.assertTrue(torch.equal(batch.state, plain.state))
        expect = torch.tensor([r.reward + float(o.lam @ o.collector.g_by_identity[(r.session_uuid, r.decision_seq)])
                               for r in rows], dtype=torch.float32)
        self.assertTrue(torch.equal(batch.reward, expect))
        self.assertTrue(torch.equal(plain.reward, torch.tensor([r.reward for r in rows], dtype=torch.float32)))

    def test_outcome_semantics_and_dual_accounting(self):
        o = orchestrator()
        diag = o.collector.diagnostics
        stoch = [d for d in diag if d["source"] == "STOCHASTIC_ACTOR"]
        self.assertEqual(len(o.dual_log), len(stoch) // O.DUAL_EVERY)
        self.assertGreaterEqual(len(o.dual_log), 1)
        # first dual update: mean over the first 400 stochastic decisions (window not yet full), excluded included
        first = o.dual_log[0]
        window = np.array([d["g"] for d in stoch[:O.DUAL_EVERY]])
        lam = np.clip(np.asarray(doc()["lambda0"]) - O.ETA * window.mean(0), 0, O.LAMBDA_MAX)
        np.testing.assert_allclose(first["lam"], lam, rtol=0, atol=1e-12)
        for d in diag:
            if d["excluded"]:
                self.assertIsNone(d["reward"])
            elif d["terminal"] == "TIMEOUT":
                self.assertEqual(d["reward"], 0.0)
            else:
                y = o.collector.catalog.reward_value(d["reward_scene_key"], mode_id=d["mode_id"], q_e4=d["q_e4"])
                self.assertEqual(d["reward"], y)
        self.assertEqual(len(o.collector.g_by_identity), sum(1 for d in diag if not d["excluded"]))

    def test_registration_rejects_drift(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "R.json"
            d = dict(doc())
            p.write_text(json.dumps(d))
            REG.verify(p, seed=43)
            for key, val in (("source_digest", "0" * 64), ("objective_sha256", "1" * 64)):
                p.write_text(json.dumps(dict(d, **{key: val})))
                with self.assertRaises(REG.RegistrationError):
                    REG.verify(p, seed=17)


if __name__ == "__main__":
    unittest.main()

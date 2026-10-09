"""Checks for the evaluation's read-only queue-order instrumentation and policy runner (CPU).

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v \
    rl_agent.splitfusion_run5_lowmcs_retrain_v1.test_evaluation
"""
from __future__ import annotations

import random
import unittest
from dataclasses import replace

from rl_agent.splitfusion_corrected_latency_env_v1 import chains as CH
from rl_agent.splitfusion_run5_full_v1 import collector as FC

from . import campaign as CAMP
from . import evaluate as EV


class EvaluationInstrumentationTest(unittest.TestCase):
    def test_instrumented_compose_is_result_neutral_and_envelope_only_raises(self):
        _, model = CAMP.components(pool="HELDOUT_GCV2")
        rng = random.Random(3)
        flips = 0
        for k in range(600):
            cid = model._ids[k % len(model._ids)]
            d = model.move(cid, ranked=model.ranked(110.0, 0.4), mode_id=11, q_e4=rng.randint(2000, 9700),
                           prior_ul_mcs=rng.randint(9, 28), backlog_bytes=0.0, wire_bytes=rng.randint(15_000, 250_000),
                           u=rng.random())
            carry = rng.choice((None, rng.uniform(100.0, 200.0)))
            real = EV._REAL_COMPOSE(d.vector, carry_in=carry)
            got = EV.instrumented_compose(d.vector, carry_in=carry)
            self.assertEqual(real, got)
            info = dict(EV.LAST)
            self.assertGreaterEqual(info["envelope"], real.receipt)
            if info["flip_gain"]:
                flips += 1
                self.assertIs(real.ordering, CH.Ordering.GT_FIRST)
                self.assertGreater(info["gain_ms"], 0.0)
            else:
                self.assertEqual(info["envelope"], real.receipt)
        self.assertGreaterEqual(flips, 0)

    def test_policy_runner_is_unchanged_by_the_instrumentation(self):
        spec = dict(kind="fixed", group="FIXED", mode=11, q=7000)
        saved = FC.compose
        try:
            a = EV.run_policy("FIXED_m11_q7000", spec, "E1", "CORRECTED", 9017, 150)
        finally:
            FC.compose = saved
        FC.compose = EV._REAL_COMPOSE
        try:
            from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
            from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
            shared, latency = CAMP.components(split="held_scene", pool="HELDOUT_GCV2")
            col = C.TrialRun5CollectorV1(latency=latency, seed=9017, shared_sources=shared, arm="C0")
            for i in range(150):
                col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=11, q_e4=7000,
                                                                   source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
            self.assertEqual([r["receipt"] for r in a], [d["receipt_ms"] for d in col.diagnostics])
            self.assertEqual([r["scene"] for r in a], [d["reward_scene_key"] for d in col.diagnostics])
        finally:
            FC.compose = saved
        self.assertTrue(all(r["envelope"] >= r["receipt"] for r in a))


if __name__ == "__main__":
    unittest.main()

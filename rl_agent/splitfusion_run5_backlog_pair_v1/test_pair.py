"""Focused integration tests (CPU)."""
from __future__ import annotations

import random
import unittest

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C

from . import env as E

SH = LAT = NEU = None


def setUpModule():
    global SH, LAT, NEU
    SH, LAT = E.components()
    _, NEU = E.components(neutralized=True)


class NeutralizedIsProductionAtZeroBacklog(unittest.TestCase):
    def test_move_identical_at_zero_backlog_and_differs_only_in_lateness_otherwise(self):
        rng = random.Random(3)
        ids = LAT._ids
        diff_pos = 0
        for _ in range(3000):
            cid = rng.choice(ids)
            ranked = LAT.ranked(LAT.chains[cid].context.camera_si, LAT.chains[cid].context.radar_p40)
            kw = dict(ranked=ranked, mode_id=11, q_e4=rng.choice([3000, 5000, 7000]), prior_ul_mcs=rng.randint(9, 28),
                      wire_bytes=rng.randint(30_000, 180_000), u=rng.random())
            a = LAT.move(cid, backlog_bytes=0, **kw)
            b = NEU.move(cid, backlog_bytes=0, **kw)
            self.assertEqual(a, b)
            B = rng.choice([500.0, 20_000.0, 120_000.0])
            a = LAT.move(cid, backlog_bytes=B, **kw)
            b = NEU.move(cid, backlog_bytes=B, **kw)
            self.assertEqual(a.delta_ms, b.delta_ms)                     # conditional latency path unchanged
            self.assertLessEqual(b.uplink_late_probability, a.uplink_late_probability + 1e-12)
            if a.uplink_late == b.uplink_late:
                self.assertEqual(a.vector, b.vector)
            else:
                diff_pos += 1
                self.assertTrue(a.uplink_late and not b.uplink_late)
                self.assertEqual(replace_e(a.vector), replace_e(b.vector))
        self.assertGreater(diff_pos, 0)


def replace_e(v):
    from dataclasses import replace
    return replace(v, E=0.0)


class DisabledReproducesControlCollector(unittest.TestCase):
    def test_disabled_equals_trial_collector(self):
        rows = []
        for cls_kw in ((C.TrialRun5CollectorV1, dict(arm=E.ARM)), (E.TrialBacklogCollectorV1, dict(backlog_transition=None))):
            cls, kw = cls_kw
            col = cls(latency=LAT, seed=17, shared_sources=SH, **kw)
            rng = random.Random(5)
            for i in range(400):
                col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=rng.randrange(12) if i % 3 else 11,
                                                                   q_e4=9700 if i % 3 else rng.choice([0, 3000, 5000]),
                                                                   source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
            rows.append((col.diagnostics, col.collector_binding_sha256, col._backlog_bytes))
        self.assertEqual(rows[0], rows[1])

    def test_enabled_uses_gcv2_chain_timing_without_substitution(self):
        bt = E.transition()
        col = E.TrialBacklogCollectorV1(latency=LAT, seed=17, shared_sources=SH, backlog_transition=bt)
        for i in range(600):
            col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=11, q_e4=3000,
                                                               source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
        self.assertTrue(all(d["base_chain_id"].startswith("gcv2#") for d in col.diagnostics))
        self.assertTrue(all(cid in bt.chains for cid in LAT._ids))
        self.assertGreater(sum(d["backlog_bytes"] > 0 for d in col.diagnostics), 0)
        self.assertEqual(len(col.current_state_features()), 22)
        print("timing flags", bt.flags)


if __name__ == "__main__":
    unittest.main(verbosity=2)

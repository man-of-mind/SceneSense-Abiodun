"""Boundaries, composition and unsupported inputs of the low-MCS uplink candidate (CPU only, existing evidence).

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m unittest -v \
    rl_agent.splitfusion_run5_uplink_lowmcs_v1.test_lowmcs
"""
from __future__ import annotations

import functools
import json
import random
from dataclasses import replace

import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
from rl_agent.splitfusion_run5_reward_trials_v1 import latency as LR
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2

from . import data as DT
from . import fit as FT
from . import latency as LC
from . import model as M
from . import plan as P

CAND = M.LowMcsCorrectionV1(14.0, 1.5, fit_wire_max=145_556, fit_mcs_min=9)


@functools.lru_cache(maxsize=None)
def transport():
    return FT.load_transport()


@functools.lru_cache(maxsize=None)
def pools():
    t = transport()
    return (LR.PoolLatencyModelV1(P.BINDING_02, t, pool=P.BASE_POOL),
            LC.CandidatePoolLatencyModelV1(P.BINDING_02, t, pool=P.BASE_POOL, correction=M.IDENTITY),
            LC.CandidatePoolLatencyModelV1(P.BINDING_02, t, pool=P.BASE_POOL, correction=CAND))


def _decisions(n=300, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        yield dict(mode_id=11, q_e4=rng.choice((4000, 5000, 6500, 8000, 9000, 9400)), prior_ul_mcs=rng.randint(9, 28),
                   backlog_bytes=rng.choice((0.0, 0.0, 0.0, 30_000.0)), wire_bytes=rng.randint(15_000, 300_000),
                   u=rng.random())


class LowMcsCandidateTest(unittest.TestCase):
    # ---------------------------------------------------------------- boundary
    def test_transport_target_is_stage_E_of_the_decomposed_chain(self):
        old, _, _ = pools()
        rows = {r["frame_id"]: r for r in DT.load_rows("gcv2")[0]}
        for cid, v in old.vectors.items():
            r = rows[old.chains[cid].frame_id]
            assert abs(v.E - r["T"]) < 1e-9                    # last send -> complete reassembly, same stamps
            assert abs(v.subsplits["E_access"] - r["access"]) < 1e-9
            assert old.chains[cid].context.wire_bytes == r["wire"]   # same byte layer as the fit rows


    def test_fit_rows_are_complete_and_uncensored(self):
        for s in P.SESSIONS:
            _, audit = DT.load_rows(s)
            assert audit["missing_stamp"] == audit["inversion"] == audit["missing_mcs"] == 0
            assert audit["receipt_rule_mismatch"] == 0


    def test_plan_is_frozen_and_candidate_constants_match(self):
        run = P.ROOT / "experiments/splitfusion_run5_uplink_lowmcs_v1/20261006_v1"
        doc = json.loads((run / "FITTING_PLAN.json").read_text())
        assert doc["plan_sha256"] == P.canonical(P.document())
        c = M.LowMcsCorrectionV1.load(run / "fit_01/CANDIDATE.json")
        assert c.provenance["plan_sha256"] == doc["plan_sha256"] and "fade" not in c.provenance["fit_sessions"]


    # ---------------------------------------------------------------- composition
    def test_identity_correction_is_bit_identical_to_binding_02(self):
        old, ident, _ = pools()
        ids = old._ids
        for i, d in enumerate(_decisions()):
            cid = ids[i % len(ids)]
            ranked = old.ranked(110.0, 0.4)
            a = old.move(cid, ranked=ranked, **d)
            b = ident.move(cid, ranked=ranked, **d)
            assert a.vector == replace(b.vector, chain_id=a.vector.chain_id)
            assert (a.uplink_late_probability, a.uplink_late, a.delta_ms) == (b.uplink_late_probability, b.uplink_late, b.delta_ms)
            assert compose(a.vector, carry_in=None) == compose(b.vector, carry_in=None)


    def test_only_stage_E_changes_and_gt_and_score_paths_are_untouched(self):
        old, _, cand = pools()
        ids = old._ids
        for i, d in enumerate(_decisions(200, seed=11)):
            cid = ids[(7 * i) % len(ids)]
            ranked = old.ranked(120.0, 0.35)
            a, b = old.move(cid, ranked=ranked, **d).vector, cand.move(cid, ranked=ranked, **d).vector
            for s in ("A", "B", "C", "D", "F", "H"):
                assert getattr(a, s) == getattr(b, s)
            assert (a.gt_segments, a.bind_end, a.costs, a.carry_in) == (b.gt_segments, b.bind_end, b.costs, b.carry_in)
            ca, cb = compose(a, carry_in=None), compose(b, carry_in=None)
            assert ca.gt_ready == cb.gt_ready                     # GT path concurrent and unchanged
            if b.E >= a.E and ca.ordering == cb.ordering:         # same score order: a later prediction is never earlier
                assert cb.receipt >= ca.receipt - 1e-9
            elif b.E >= a.E and cb.receipt < ca.receipt - 1e-9:
                # disclosed property of the UNCHANGED SERVICE_QUEUE_V2: a later prediction can flip the score order to
                # GT_FIRST, whose measured role costs differ; this is the only way a receipt gets earlier
                assert (ca.ordering.value, cb.ordering.value) == ("PREDICTION_FIRST", "GT_FIRST")


    def test_base_action_reproduces_the_measured_chain(self):
        _, _, cand = pools()
        for cid in cand._ids:
            ctx = cand.chains[cid].context
            d = cand.move(cid, ranked=cand.ranked(ctx.camera_si, ctx.radar_p40), mode_id=ctx.mode_id, q_e4=ctx.q_e4,
                          prior_ul_mcs=ctx.prior_ul_mcs, backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes, u=1.0)
            assert d.delta_ms["E"] == 0.0 and d.vector.E == max(cand.vectors[cid].E, cand.vectors[cid].subsplits["E_access"])


    def test_high_mcs_base_and_decision_identical_to_old(self):
        old, _, cand = pools()
        high = [c for c in old._ids if old.chains[c].context.prior_ul_mcs >= 16]
        for i, d in enumerate(_decisions(300, seed=3)):
            if d["prior_ul_mcs"] < 16:
                continue
            cid = high[i % len(high)]
            ranked = old.ranked(100.0, 0.5)
            a, b = old.move(cid, ranked=ranked, **d), cand.move(cid, ranked=ranked, **d)
            assert a.vector == replace(b.vector, chain_id=a.vector.chain_id)


    def test_lambda_scales_only_the_zero_mean_residual(self):
        _, _, cand = pools()
        import numpy as np
        res = [cand.vectors[c].E - cand._base_mu[c] - cand.beta_pool for c in cand._ids]
        assert abs(float(np.mean(res))) < 1e-9


    # ---------------------------------------------------------------- unsupported inputs
    def test_positive_backlog_at_low_mcs_is_unresolved_not_corrected(self):
        assert CAND.term_ms(backlog_bytes=10.0, wire_bytes=150_000, prior_ul_mcs=9) == 0.0
        assert CAND.spread(backlog_bytes=10.0, prior_ul_mcs=9) == 1.0
        assert CAND.flags(backlog_bytes=10.0, wire_bytes=150_000, prior_ul_mcs=9) == (M.UNRESOLVED_BACKLOG,)
        _, _, cand = pools()
        cid = cand._ids[0]
        d = cand.move(cid, ranked=cand.ranked(110.0, 0.4), mode_id=11, q_e4=5000, prior_ul_mcs=9, backlog_bytes=20_000.0,
                      wire_bytes=120_000, u=1.0)
        assert M.UNRESOLVED_BACKLOG in d.floors


    def test_extrapolation_is_flagged_never_silent(self):
        assert M.EXTRAPOLATED_PAYLOAD in CAND.flags(backlog_bytes=0.0, wire_bytes=200_000, prior_ul_mcs=10)
        assert M.EXTRAPOLATED_MCS in CAND.flags(backlog_bytes=0.0, wire_bytes=50_000, prior_ul_mcs=8)
        assert CAND.flags(backlog_bytes=0.0, wire_bytes=100_000, prior_ul_mcs=12) == ()
        assert CAND.flags(backlog_bytes=5e5, wire_bytes=600_000, prior_ul_mcs=20) == ()     # high MCS: T_v2b's own support


    def test_t_v2b_support_refusal_is_preserved(self):
        _, _, cand = pools()
        with self.assertRaises(A2.OutOfSupport):
            cand.mu(backlog_bytes=0.0, wire_bytes=700_000, prior_ul_mcs=9)
        with self.assertRaises(A2.OutOfSupport):
            cand.mu(backlog_bytes=0.0, wire_bytes=50_000, prior_ul_mcs=7)


    def test_correction_is_monotone_and_zero_at_or_above_reference_mcs(self):
        prev = -1.0
        for w in range(6_000, 400_000, 5_000):
            t = CAND.term_ms(backlog_bytes=0.0, wire_bytes=w, prior_ul_mcs=9)
            assert t >= prev
            prev = t
        for w in (50_000, 150_000):
            assert [CAND.term_ms(backlog_bytes=0.0, wire_bytes=w, prior_ul_mcs=m) for m in range(9, 17)] == sorted(
                [CAND.term_ms(backlog_bytes=0.0, wire_bytes=w, prior_ul_mcs=m) for m in range(9, 17)], reverse=True)
        assert all(CAND.term_ms(backlog_bytes=0.0, wire_bytes=300_000, prior_ul_mcs=m) == 0.0 for m in range(16, 29))
        assert CAND.term_ms(backlog_bytes=0.0, wire_bytes=P.W0_BYTES, prior_ul_mcs=9) == 0.0


    def test_invalid_parameters_and_foreign_artifacts_refused(self):
        with self.assertRaises(ValueError):
            M.LowMcsCorrectionV1(-1.0, 1.0, 1, 9)
        with self.assertRaises(ValueError):
            M.LowMcsCorrectionV1(1.0, 0.0, 1, 9)
        bad = dict(CAND.to_dict(), w0=1)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "c.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                M.LowMcsCorrectionV1.load(Path(tmp) / "c.json")


if __name__ == "__main__":
    unittest.main()

"""Focused CPU tests for the restricted Run-5 pilot (state, support, reward/outcome, timing, fallback, trainer).

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m unittest -v \
    rl_agent.splitfusion_run5_restricted_pilot_v1.test_pilot
"""
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR
from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM

from . import authorization as A, collector as C, latency as L, quality as Q, support as S, trainer as TR

BINDING = Path(__file__).resolve().parents[2] / "experiments/splitfusion_run5_restricted_pilot_v1/binding_01"
torch.set_num_threads(4)


class Fixture:
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            auth = Path(tempfile.mkdtemp(prefix="pilot_auth_")) / "AUTH.json"
            A.issue(auth, binding_dir=BINDING, authorized_by="TEST", authorized_on="TEST", scope="unit tests only")
            lat = L.PilotLatencyModelV1(BINDING, "TRAIN")
            cat = Q.PilotSceneCatalogV1("fit", lat.scene_supported)
            trace = [t for t in json.loads((BINDING / "SPLIT.json").read_text())["trace"] if t["split"] == "TRAIN"]
            fb = C.FallbackModelV1(json.loads((BINDING / "FALLBACK.json").read_text()))
            cls._cache = dict(auth=auth, lat=lat, cat=cat, trace=trace, fb=fb, shared=CR.load_shared_sources())
        return cls._cache


def collector(arm="CONTROL", seed=17, shift=0.0):
    f = Fixture.get()
    return C.PilotRun5CollectorV1(latency=f["lat"], catalog=f["cat"], trace=f["trace"], fallback=f["fb"], arm=arm,
                                  seed=seed, authorization=f["auth"], binding_dir=BINDING, shared_sources=f["shared"],
                                  receipt_shift_ms=shift)


def run_decisions(col, n=250, seed=5):
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
    rng = random.Random(seed)
    out = []
    for i in range(n):
        mode = rng.choice(S.ALLOWED_MODES)
        lo, hi = S.BRACKETS[mode]
        q = rng.choice([rng.randint(lo, hi)] + list(S.MEASURED_POINTS[mode]))
        req = orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q, source="STOCHASTIC_ACTOR",
                                          warmup_q_bin_index=None)
        out.append((col.collect_transition(req), col.diagnostics[-1]))
    return out


class StateTest(unittest.TestCase):
    def test_22d_state_snr_and_previous_q_perc_unchanged_in_candidate_arm(self):
        col = collector("CANDIDATE")
        rows = run_decisions(col, 120)
        checked = False
        for i, (tr, d) in enumerate(rows):
            state = d["state"]
            self.assertEqual(len(state), 22)
            snr_scaled = min(1.0, max(0.0, (d["snr_db"] - 5.5) / 19.0))
            self.assertAlmostEqual(state[21], snr_scaled, places=6)
            if i and tr is not None and not rows[i - 1][1]["excluded"] and rows[i - 1][1]["terminal"] == "SUCCESS":
                self.assertAlmostEqual(state[17], rows[i - 1][1]["q_perc"], places=12)   # previous Q_perc, not reward
                if abs(rows[i - 1][1]["q_perc"] - rows[i - 1][1]["candidate_reward"]) > 1e-6:
                    self.assertNotAlmostEqual(state[17], rows[i - 1][1]["candidate_reward"], places=6)
                    checked = True
        self.assertTrue(checked)

    def test_channel_triplet_comes_from_one_measured_decision(self):
        f = Fixture.get()
        by = {(t["source_id"], t["frame_id"]): t for t in f["trace"]}
        for _, d in run_decisions(collector(), 60):
            t = by[tuple(d["trace_entry"])]
            self.assertEqual((d["prior_ul_mcs"], d["snr_db"], d["backlog_bytes"]),
                             (t["prior_ul_mcs"], t["snr_db"], t["backlog_bytes"]))


class SupportTest(unittest.TestCase):
    def test_masked_modes_have_zero_probability_and_q_in_brackets(self):
        actor = S.PilotRestrictedActor(RM.run5_model_config())
        x = torch.randn(256, 22)
        s = actor.sample_all_modes(x, generator=torch.Generator().manual_seed(3))
        masked = [m for m in range(12) if m not in S.ALLOWED_MODES]
        self.assertEqual(float(s.probs[:, masked].detach().max()), 0.0)
        self.assertTrue(torch.isfinite(s.log_prob_discrete).all() and torch.isfinite(s.log_prob_continuous).all())
        for m in S.ALLOWED_MODES:
            lo, hi = S.BRACKETS[m]
            self.assertTrue(lo <= int(s.q_e4[:, m].min()) and int(s.q_e4[:, m].max()) <= hi)
        d = actor.deterministic_execution(x)
        self.assertTrue(all(S.in_support(int(m), int(q)) for m, q in zip(d.mode_index, d.q_e4)))
        entropy = float((-(s.probs * s.log_prob_discrete).sum(-1)).max())
        self.assertLessEqual(entropy, math.log(3) + 1e-6)

    def test_brackets_inside_frozen_actor_support_and_warmup(self):
        for m, (lo, hi) in S.BRACKETS.items():
            olo, ohi = S.ORIGINAL_BOUNDS[m]
            self.assertTrue(olo <= lo and hi <= ohi)
        acts, _ = S.warmup_actions(17, 6, TR.WARMUP_SAMPLES_PER_MODE_Q_BIN)
        self.assertEqual(len(acts), 288)
        self.assertEqual({a.mode_id for a in acts}, set(S.ALLOWED_MODES))
        self.assertTrue(all(S.in_support(a.mode_id, a.q_e4) for a in acts))

    def test_original_actor_refuses_pilot_checkpoint_and_vice_versa(self):
        pilot = S.PilotRestrictedActor(RM.run5_model_config())
        original, _ = RM.build_run5_models(actor_seed=1, critic_seed=2)
        with self.assertRaises(Exception):
            original.load_state_dict(pilot.state_dict())
        with self.assertRaises(Exception):
            S.PilotRestrictedActor(RM.run5_model_config()).load_state_dict(original.state_dict())

    def test_collector_refuses_actions_outside_support(self):
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
        col = collector()
        for mode, q in ((11, 9000), (5, 6000), (2, 9800), (8, 4000)):
            with self.assertRaises(C.PilotCollectorError):
                col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=0, mode_id=mode, q_e4=q,
                                                                   source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))


class RewardOutcomeTest(unittest.TestCase):
    def test_reward_arithmetic_and_outcome_semantics(self):
        f = Fixture.get()
        for arm in ("CONTROL", "CANDIDATE"):
            seen = set()
            for tr, d in run_decisions(collector(arm), 300):
                timely = d["receipt_ms"] is not None and d["receipt_ms"] <= 170.0
                if d["excluded"]:
                    self.assertTrue(timely and d["q_perc"] is None)          # timely undefined -> excluded
                    self.assertIsNone(tr)
                    seen.add("EXCLUDED")
                    continue
                if d["terminal"] == "SUCCESS":
                    self.assertTrue(timely and d["q_perc"] is not None)
                    self.assertAlmostEqual(d["control_reward"], d["q_perc"] - 0.25 * d["receipt_ms"] / 170.0, places=6)
                    cand = f["cat"].candidate(d["reward_scene_key"], mode_id=d["mode_id"], q_e4=d["q_e4"])
                    self.assertEqual(d["candidate_reward"], cand)
                else:
                    self.assertEqual(d["terminal"], "TIMEOUT")
                    self.assertFalse(timely)
                    self.assertEqual((d["control_reward"], d["candidate_reward"]), (-1.0, 0.0))
                    if d["q_perc"] is None:
                        seen.add("LATE_UNDEFINED_TIMEOUT")
                self.assertEqual(tr.reward, d["control_reward"] if arm == "CONTROL" else d["candidate_reward"])
                seen.add(d["terminal"])
            self.assertTrue({"SUCCESS", "TIMEOUT", "EXCLUDED"} <= seen)

    def test_candidate_definedness_and_formula_on_grid_rows(self):
        cat = Fixture.get()["cat"]
        for key in cat.keys[:60]:
            for (mode, q), cell in cat._scenes[key]["grid"].items():
                self.assertEqual(cell["candidate"] is None, cell["q_perc"] is None)
                if cell["candidate"] is not None:
                    comp = cell["components"]
                    num = den = 0.0
                    for cls, w in (("person", 0.6), ("vehicle", 0.4)):
                        if comp[cls]["eligible"]:
                            num += w * comp[cls]["hits"] / comp[cls]["eligible"]
                            den += w
                    self.assertGreaterEqual(cell["candidate"], 0.0)
                    self.assertLessEqual(cell["candidate"], num / den + 1e-12)
        key = cat.defined_keys[0]
        a = cat.candidate(key, mode_id=11, q_e4=7000)
        b = cat.candidate(key, mode_id=11, q_e4=8000)
        self.assertAlmostEqual(cat.candidate(key, mode_id=11, q_e4=7500), (a + b) / 2, places=12)


class TimingTest(unittest.TestCase):
    def test_measured_chains_recompose_exactly(self):
        lat = Fixture.get()["lat"]
        worst = 0.0
        for cid, chain in lat.chains.items():
            comp = compose(lat.vectors[cid])
            for name, value in comp.instants.items():
                if chain.instants.get(name) is not None:
                    worst = max(worst, abs(value - chain.instants[name]))
        self.assertLess(worst, 1e-9)
        self.assertEqual(set(lat.rejected), set(L.REGISTERED_REJECTIONS))

    def test_interpolation_changes_only_action_dependent_stages(self):
        lat = Fixture.get()["lat"]
        for q in (3500, 5000, 7500):
            d = lat.draw(mode_id=11, q_e4=q, profile=S.PROFILE, camera_si=116.0, radar_p40=0.39, prior_ul_mcs=26,
                         rng=random.Random(4))
            base = lat.vectors[d.chain_id]
            partner = lat.vectors[d.partner_chain_id]
            lo_v, hi_v = (base, partner) if d.endpoints[0] == lat.chains[d.chain_id].context.q_e4 else (partner, base)
            for stage in ("A", "B", "H", "R", "gt_segments", "costs", "dispatch_prediction", "dispatch_target",
                          "post_span", "carry_in", "bind_end"):
                self.assertEqual(getattr(d.vector, stage), getattr(base, stage) if stage != "costs" else d.vector.costs)
            for stage in L.INTERPOLATED_STAGES:
                expected = (1 - d.weight) * getattr(lo_v, stage) + d.weight * getattr(hi_v, stage)
                self.assertAlmostEqual(getattr(d.vector, stage), expected, places=9)
            composed = compose(d.vector)
            self.assertIsNotNone(composed.ordering)

    def test_one_random_draw_per_decision(self):
        lat = Fixture.get()["lat"]
        for mode, q in ((11, 3000), (11, 5500), (5, 8000), (8, 9800)):
            r1, r2 = random.Random(9), random.Random(9)
            lat.draw(mode_id=mode, q_e4=q, profile=S.PROFILE, camera_si=116.0, radar_p40=0.39, prior_ul_mcs=26, rng=r1)
            r2.randrange(lat.k)
            self.assertEqual(r1.getstate(), r2.getstate())

    def test_receipt_shift_moves_only_the_receipt(self):
        a, b = collector(shift=0.0), collector(shift=5.0)
        ra, rb = run_decisions(a, 40), run_decisions(b, 40)
        for (_, x), (_, y) in zip(ra[:1], rb[:1]):
            self.assertAlmostEqual(y["receipt_ms"] - x["receipt_ms"], 5.0, places=9)
            self.assertEqual(x["prediction_at_service_ms"], y["prediction_at_service_ms"])


class FallbackAndCompatibilityTest(unittest.TestCase):
    def test_fallback_is_causal_and_geometric(self):
        fb = Fixture.get()["fb"]
        self.assertEqual(fb.probability(8, 9800), fb.p[(8, 9800)])
        self.assertAlmostEqual(fb.probability(11, 7500), (fb.p[(11, 7000)] + fb.p[(11, 8000)]) / 2)
        self.assertEqual(C.FallbackModelV1.frames(0.1, 0.5), 0)
        self.assertEqual(C.FallbackModelV1.frames(0.1, 0.05), 1)
        self.assertEqual(C.FallbackModelV1.frames(0.1, 0.005), 2)
        col = collector()
        rows = run_decisions(col, 80)
        for _, d in rows:
            self.assertIsNotNone(d["fallback_frames_after"])          # drawn after the decision executed
        self.assertEqual(col.fallback_frames_total, C.STARTUP_FALLBACKS + sum(d["fallback_frames_after"] for _, d in rows))

    def test_runtime_compatibility_and_source_identity(self):
        compat = json.loads((BINDING / "COMPATIBILITY.json").read_text())
        self.assertTrue(compat["compatible"])
        lat = Fixture.get()["lat"]
        self.assertEqual({c.source_id for c in lat.stratum(11, 3000)}, {"v2_a67_m11_q3000", "v3_a67_m11_q3000"})


class SessionIdentityTest(unittest.TestCase):
    def test_session_uuid_valid_and_unique_beyond_1000_and_unchanged_below(self):
        import uuid
        col = collector()
        seen = set()
        for index in (0, 1, 999, 1000, 1001, 4095, 70000):
            col._session_index = index
            value = col._session_uuid()
            self.assertEqual(str(uuid.UUID(value)), value)
            seen.add(value)
            if index < 1000:
                self.assertEqual(value, CR.CorrectedRun4CollectorV1._session_uuid(col))
        self.assertEqual(len(seen), 7)


class TrainerTest(unittest.TestCase):
    def test_finite_updates_and_bundle_restore_is_exact(self):
        f = Fixture.get()
        factory = lambda: collector("CANDIDATE", seed=17)
        o = TR.PilotOrchestratorV1(collector_factory=factory, seed=17, arm="CANDIDATE", authorization_sha256="test")
        o.warmup_phase()
        for _ in range(3):
            o.step_update()
        path = Path(tempfile.mkdtemp(prefix="pilot_bundle_")) / "u3.pt"
        o.save(path)
        r = TR.PilotOrchestratorV1.restore(path, collector_factory=factory, seed=17, arm="CANDIDATE",
                                           authorization_sha256="test")
        self.assertEqual(r.state_digest(), o.state_digest())
        m1 = [o.step_update().critic_loss for _ in range(2)]
        m2 = [r.step_update().critic_loss for _ in range(2)]
        self.assertEqual(m1, m2)
        self.assertEqual(r.state_digest(), o.state_digest())


if __name__ == "__main__":
    unittest.main()

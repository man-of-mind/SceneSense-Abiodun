"""Focused CPU checks for the reward trials: reward boundaries/undefined cases, action support, causal features,
latency accounting (binding_02 pools), checkpoint resume, authorization binding.

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m unittest -v \
    rl_agent.splitfusion_run5_reward_trials_v1.test_trials
"""
import collections
import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
from rl_agent.splitfusion_corrected_latency_env_v1.scene_catalog import FitSceneCatalogWithUndefinedV1
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import ConditionalHybridActor
from rl_agent.splitfusion_run5_full_v1 import latency as LF
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_restricted_pilot_v1.quality import candidate_from_row
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import authorization as A, campaign as CAMP, catalog as K, collector as C, evidence as EV, latency as L
from . import rewards as R, trainer as TR

ROOT = Path(__file__).resolve().parents[2]
BINDING = CAMP.BINDING
torch.set_num_threads(4)


class Fixture:
    _cache = {}

    @classmethod
    def get(cls, arm="C1", split="fit", pool="TRAIN_GCV2"):
        key = (arm, split, pool)
        if key not in cls._cache:
            cls._cache[key] = CAMP.components(arm, split=split, pool=pool)
        return cls._cache[key]


def collector(seed=17, arm="C1", split="fit", pool="TRAIN_GCV2"):
    shared, lat = Fixture.get(arm, split, pool)
    return C.TrialRun5CollectorV1(latency=lat, seed=seed, shared_sources=shared, arm=arm)


def random_decisions(col, n, seed=5):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        mode = rng.randrange(12)
        lo, hi = S.BOUNDS[mode]
        q = rng.choice([lo, hi, rng.randint(lo, hi)])
        req = orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q, source="STOCHASTIC_ACTOR",
                                          warmup_q_bin_index=None)
        out.append((col.collect_transition(req), col.diagnostics[-1]))
    return out


def all_rows():
    con = sqlite3.connect(f"file:{ROOT / V1.PAYLOAD_AUTHORITY_RELPATH}?mode=ro&immutable=1", uri=True)
    rows = [json.loads(b) for (b,) in con.execute("SELECT row_json FROM quality_rows")]
    con.close()
    return rows


def synthetic(v=None, p=None, seg=0.5):
    """v/p = (hits, eligible) with hits matched inside tolerance; None = class not eligible."""
    row = {"q_seg": seg}
    defined = False
    for cls, spec, tol in (("vehicle", v, 1.0), ("person", p, 1.2)):
        hits, eligible = spec if spec else (0, 0)
        row[f"loc_{cls}_eligible_gt"] = eligible
        row[f"loc_{cls}_matched_xy_errors_m"] = [tol] * hits + [tol + 0.01]     # boundary hit + one beyond
        defined = defined or eligible > 0
    row["q_perc"] = 0.5 if defined else None
    return row


class RewardTest(unittest.TestCase):
    def test_c0_is_bit_identical_to_the_historical_reward_and_definedness_matches_q_perc(self):
        n = collections.Counter()
        for row in all_rows():
            want = candidate_from_row(row)
            values = {a: R.value_from_row(row, a) for a in R.ARMS}
            self.assertEqual(values["C0"], want)                                   # exact, incl. None
            for a in R.ARMS:
                self.assertEqual(values[a] is None, row["q_perc"] is None)
            eligible = [c for c in R.TOL if row[f"loc_{c}_eligible_gt"]]
            if len(eligible) == 1:
                self.assertEqual(values["C2"], values["C1"])                       # M = Y exactly
                n["one_class"] += 1
            elif len(eligible) == 2:
                n["two_class"] += 1
            else:
                n["undefined"] += 1
        self.assertGreater(min(n["one_class"], n["two_class"], n["undefined"]), 100, n)

    def test_hand_computed_values_boundaries_and_neutral_segmentation(self):
        row = synthetic(v=(2, 4), p=(1, 2), seg=0.5)                             # Rv 0.5, Rp 0.5 -> Y 0.5, M 0.5
        self.assertAlmostEqual(R.value_from_row(row, "C0"), 0.5 * 0.85, places=15)
        self.assertAlmostEqual(R.value_from_row(row, "C1"), 0.5 * 0.95, places=15)
        row = synthetic(v=(4, 4), p=(1, 4), seg=0.0)                             # Rv 1, Rp .25 -> Y .55, M .25
        self.assertAlmostEqual(R.value_from_row(row, "C0"), 0.55 * 0.7, places=15)
        self.assertAlmostEqual(R.value_from_row(row, "C1"), 0.55 * 0.9, places=15)
        self.assertAlmostEqual(R.value_from_row(row, "C2"), (0.75 * 0.55 + 0.25 * 0.25) * 0.9, places=15)
        row = synthetic(v=(3, 3), p=None, seg=None)                              # one class, S undefined
        self.assertAlmostEqual(R.value_from_row(row, "C0"), 1.0, places=15)     # historical float: 0.4*3/3/0.4
        self.assertEqual(R.value_from_row(row, "C1"), R.value_from_row(row, "C0"))   # S undefined -> neutral
        self.assertEqual(R.value_from_row(row, "C2"), R.value_from_row(row, "C1"))
        for arm in R.ARMS:
            self.assertIsNone(R.value_from_row(synthetic(v=None, p=None), arm))    # no min(empty), no invented Q
        row = synthetic(v=(0, 2), p=(0, 3), seg=1.0)                             # all misses -> 0, not undefined
        self.assertEqual([R.value_from_row(row, a) for a in R.ARMS], [0.0, 0.0, 0.0])
        bad = synthetic(v=(1, 1)); bad["q_perc"] = None
        with self.assertRaises(R.RewardError):
            R.value_from_row(bad, "C1")
        self.assertEqual(len({R.reward_sha256(a) for a in R.ARMS}), 3)

    def test_off_grid_value_is_linear_between_anchors_and_undefined_if_either_is(self):
        shared, _ = Fixture.get("C2")
        cat = shared["catalog"]
        for key in cat.keys[:60] + cat.partially_undefined_keys[:10]:
            for mode in (5, 11):
                lo, hi = 5000, 6000
                a = cat.value_for("C2", key, mode_id=mode, q_e4=lo)
                b = cat.value_for("C2", key, mode_id=mode, q_e4=hi)
                got = cat.reward_value(key, mode_id=mode, q_e4=5250)
                if a is None or b is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, a + 0.25 * (b - a), places=12)
                self.assertEqual(got is None, cat.draw(key, mode_id=mode, q_e4=5250)[0] is None)

    def test_trial_fit_catalogue_equals_the_corrected_fit_catalogue(self):
        cat = Fixture.get("C0")[0]["catalog"]
        ref = FitSceneCatalogWithUndefinedV1()
        self.assertEqual((cat.keys, cat.binding_sha256), (ref.keys, ref.binding_sha256))
        for key in cat.keys[::7]:
            for mode in range(12):
                for q in (0, 1500, 3333, 9800):
                    self.assertEqual(cat.draw(key, mode_id=mode, q_e4=q), ref.draw(key, mode_id=mode, q_e4=q))

    def test_outcome_rewards_session_reset_and_late_undefined(self):
        col = collector(17, arm="C2")
        rows = random_decisions(col, 1500)
        terminals = collections.Counter(d["terminal"] for _, d in rows)
        self.assertGreater(terminals["EVALUATOR_FAULT"], 0)
        sessions = 1
        cat = col.catalog
        for tr, d in rows:
            if d["terminal"] == "SUCCESS":
                self.assertEqual(tr.reward, cat.value_for("C2", d["reward_scene_key"], mode_id=d["mode_id"], q_e4=d["q_e4"]))
                self.assertLessEqual(d["receipt_ms"], 170.0)
            elif d["terminal"] == "TIMEOUT":
                self.assertEqual(tr.reward, 0.0)
                self.assertTrue(d["receipt_ms"] is None or d["receipt_ms"] > 170.0)
            else:
                self.assertIsNone(tr)
                self.assertTrue(d["excluded"] and d["q_perc"] is None and d["receipt_ms"] <= 170.0)
                sessions += 1
        self.assertEqual(col._session_index + 1, sessions)
        self.assertTrue(all(not d["excluded"] for _, d in rows if d["terminal"] == "TIMEOUT" and d["q_perc"] is None))


class SupportAndStateTest(unittest.TestCase):
    def test_actor_original_class_bounds_no_mask_and_warmup(self):
        o = TR.TrialOrchestratorV1(collector_factory=lambda: collector(17, "C0"), seed=17, authorization_sha256="TEST")
        self.assertIs(type(o.actor), ConditionalHybridActor)
        self.assertFalse(any("mask" in name for name, _ in o.actor.named_buffers()))
        states = torch.tensor([d["state"] for _, d in random_decisions(collector(), 40)], dtype=torch.float32)
        with torch.no_grad():
            sample = o.actor.sample_all_modes(states, generator=torch.Generator().manual_seed(0))
        self.assertTrue(bool((sample.probs > 0).all()))
        for m, (lo, hi) in enumerate(S.BOUNDS):
            self.assertTrue(bool(((sample.q_e4[:, m] >= lo) & (sample.q_e4[:, m] <= hi)).all()))
        cells = collections.Counter((a.mode_id, a.q_bin_index) for a in
                                    (o.schedule.action_at(i) for i in range(len(o.schedule))))
        self.assertEqual(set(cells), {(m, b) for m in range(12) for b in range(6)})
        self.assertTrue(all(v == 4 for v in cells.values()))
        self.assertEqual(o.binding_document["reward_sha256"], R.reward_sha256("C0"))
        bad = orch.ModeledActionRequestV1(decision_ordinal=0, mode_id=0, q_e4=5000, source="STOCHASTIC_ACTOR",
                                          warmup_q_bin_index=None)
        with self.assertRaises(C.TrialCollectorError):
            collector().collect_transition(bad)

    def test_training_refuses_held_scenes_and_evaluation_pools(self):
        for split, pool in (("held_scene", "TRAIN_GCV2"), ("fit", "HELDOUT_GCV2"), ("fit", "RETENTION_V1")):
            with self.assertRaises(TR.TrialTrainingError):
                TR.TrialOrchestratorV1(collector_factory=lambda: collector(17, "C1", split, pool), seed=17,
                                       authorization_sha256="TEST")

    def test_22d_causal_snr_previous_q_perc_not_the_reward(self):
        rows = random_decisions(collector(29, arm="C1"), 650)
        profiles, checked = set(), 0
        for i, (tr, d) in enumerate(rows):
            state = d["state"]
            self.assertEqual(len(state), 22)
            self.assertAlmostEqual(state[21], min(1.0, max(0.0, (d["snr_db"] - 5.5) / 19.0)), places=6)
            self.assertAlmostEqual(state[2], d["prior_ul_mcs"] / 28.0, places=6)
            profiles.add(d["profile"])
            prev = rows[i - 1][1] if i else None
            if prev and not prev["excluded"] and not d["excluded"] and prev["terminal"] == "SUCCESS":
                self.assertAlmostEqual(state[17], prev["q_perc"], places=12)
                if abs(prev["q_perc"] - prev["reward_value"]) > 1e-6:
                    checked += 1
        self.assertGreater(checked, 20)
        self.assertEqual(profiles, {"FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"})

    def test_state_is_observed_before_the_action_and_scene_stream_is_action_independent(self):
        a, b = collector(43, "C0"), collector(43, "C2")
        before = a.current_state_features()
        random_decisions(a, 30, seed=1)
        random_decisions(b, 30, seed=2)
        self.assertEqual(a.diagnostics[0]["state"], before)
        self.assertEqual([d["reward_scene_key"] for d in a.diagnostics], [d["reward_scene_key"] for d in b.diagnostics])
        self.assertEqual([d["profile"] for d in a.diagnostics], [d["profile"] for d in b.diagnostics])


class LatencyTest(unittest.TestCase):
    def test_pools_are_disjoint_deterministic_and_inherit_binding_01(self):
        doc = json.loads((BINDING / "LATENCY_EVIDENCE.json").read_text())
        old = json.loads((EV.BINDING_01 / "LATENCY_EVIDENCE.json").read_text())
        for section in ("structure_288", "transfer", "uplink"):
            self.assertEqual(doc[section], old[section])
        train, held = set(doc["pools"]["TRAIN_GCV2"]["chain_ids"]), set(doc["pools"]["HELDOUT_GCV2"]["chain_ids"])
        self.assertFalse(train & held)
        rows = EV.decision_index_by_frame(EV.SOURCES["gcv2"]["rows"])
        for cid in train | held:
            block = EV.block_of(rows[int(cid.split("#")[1])])
            self.assertEqual(block, "TRAIN" if cid in train else "HELDOUT")
        self.assertTrue(all(c.startswith("retention#") for c in doc["pools"]["RETENTION_V1"]["chain_ids"]))
        self.assertFalse(any("gc_v2" in c or "retention" in c for c in old["transfer"]["capture_profiles"]))

    def test_zero_delta_reproduces_every_measured_chain_exactly_in_every_pool(self):
        for pool in EV.POOLS:
            _, lat = Fixture.get("C1", "fit", pool)
            for cid, chain in lat.chains.items():
                ctx = chain.context
                d = lat.move(cid, ranked=lat.ranked(ctx.camera_si, ctx.radar_p40), mode_id=ctx.mode_id, q_e4=ctx.q_e4,
                             prior_ul_mcs=ctx.prior_ul_mcs, backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes,
                             u=1.0)
                self.assertEqual((d.delta_ms["C"], d.delta_ms["D"], d.delta_ms["F"]), (0.0, 0.0, 0.0))
                self.assertAlmostEqual(d.delta_ms["E"], 0.0, places=12)
                self.assertFalse(d.uplink_late)
                self.assertAlmostEqual(compose(d.vector, carry_in=chain.service.carry_in).receipt,
                                       chain.instants["ue_receipt"], places=6, msg=cid)

    def test_every_stage_once_gt_concurrent_and_common_random_numbers(self):
        rows = random_decisions(collector(17), 300)
        for _, d in rows:
            pred = sum(d["stage_ms"][s] for s in "ABCDEFH")
            self.assertAlmostEqual(pred, d["prediction_at_service_ms"], places=6)
            self.assertGreaterEqual(d["receipt_ms"], max(pred, d["gt_ready_ms"]))
            self.assertLess(d["receipt_ms"], pred + d["gt_ready_ms"])
            self.assertTrue(d["base_chain_id"].startswith("gcv2#"))
        _, lat = Fixture.get()
        held = set(json.loads((BINDING / "LATENCY_EVIDENCE.json").read_text())["pools"]["HELDOUT_GCV2"]["chain_ids"])
        self.assertFalse(held & {d["base_chain_id"] for _, d in rows})
        r1, r2 = random.Random(9), random.Random(9)
        lat.draw(mode_id=0, q_e4=9800, camera_si=10.0, radar_p40=0.2, prior_ul_mcs=20, backlog_bytes=0.0,
                 wire_bytes=70_000, rng=r1)
        lat.draw(mode_id=11, q_e4=500, camera_si=10.0, radar_p40=0.2, prior_ul_mcs=12, backlog_bytes=5_000.0,
                 wire_bytes=200_000, rng=r2)
        self.assertEqual(r1.getstate(), r2.getstate())
        for name in ("draw", "ranked", "_with_donor", "delta"):                    # composition logic inherited
            self.assertIs(getattr(type(lat), name), getattr(LF.FullSupportLatencyModelV1, name))

    def test_confirm_send_gate_reproduces_the_gated_chain_and_delays_gt_for_late_sends(self):
        _, lat = Fixture.get()
        doc = lat.document
        gated = doc["sources"]["gcv2"]["send_gated_chains"]
        self.assertEqual(len(gated), 1)
        cid = next(iter(gated))
        self.assertIn(cid, lat.chains)
        self.assertAlmostEqual(lat.confirm_gate_ms, gated[cid]["send_to_confirm_ms"], places=12)
        ctx = lat.chains[cid].context
        ranked = lat.ranked(ctx.camera_si, ctx.radar_p40)
        d = lat.move(cid, ranked=ranked, mode_id=ctx.mode_id, q_e4=ctx.q_e4, prior_ul_mcs=ctx.prior_ul_mcs,
                     backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes, u=1.0)
        self.assertIn("CONFIRM_SEND_AFTER_BIND", d.floors)
        self.assertAlmostEqual(compose(d.vector, carry_in=lat.chains[cid].service.carry_in).receipt,
                               lat.chains[cid].instants["ue_receipt"], places=6)
        other = sorted(c for c in lat.chains if c != cid)[0]                      # ungated chain, forced late send
        octx = lat.chains[other].context
        base = lat.move(other, ranked=lat.ranked(octx.camera_si, octx.radar_p40), mode_id=octx.mode_id,
                        q_e4=octx.q_e4, prior_ul_mcs=octx.prior_ul_mcs, backlog_bytes=octx.backlog_bytes,
                        wire_bytes=octx.wire_bytes, u=1.0)
        v = base.vector
        push = v.bind_end + dict(v.gt_segments)["gt_confirm_send_dispatch"] - (v.A + v.B + v.C + v.D) + 20.0
        from dataclasses import replace as _replace
        import unittest.mock as mock
        stretched = _replace(base, vector=_replace(v, C=v.C + push))
        with mock.patch.object(LF.FullSupportLatencyModelV1, "move", return_value=stretched):
            g = lat.move(other, ranked=[], mode_id=11, q_e4=5000, prior_ul_mcs=20, backlog_bytes=0.0,
                         wire_bytes=100_000, u=1.0)
        self.assertIn("CONFIRM_SEND_GATED", g.floors)
        self.assertAlmostEqual(g.vector.G - v.G, 20.0 + lat.confirm_gate_ms, places=6)
        compose(g.vector)                                                          # no UncalibratedTermError


class EvaluationRuleTest(unittest.TestCase):
    def test_offline_deterministic_rule_equals_the_live_adapter_on_the_historical_actor(self):
        from rl_agent.splitfusion_run5_full_live_v1 import actor as LIVE
        from . import evaluate as E
        module = E.load_actor(ROOT / E.HISTORICAL["path"])
        live = LIVE.FrozenFullRun5Actor(E.load_actor(ROOT / E.HISTORICAL["path"]))
        col = collector(9017, "C0", "held_scene", "HELDOUT_GCV2")
        for i in range(200):
            features = col.current_state_features()
            mode, q = E.act(module, features)
            decision = live.act_on_vector(tuple(features))
            self.assertEqual((mode, q), (decision.mode_id, decision.q_e4))
            col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=mode, q_e4=q,
                                                               source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))


class CheckpointTest(unittest.TestCase):
    def test_bundle_restore_is_digest_equal_and_continues_bit_identically(self):
        factory = lambda: collector(17, "C2")
        o = TR.TrialOrchestratorV1(collector_factory=factory, seed=17, authorization_sha256="TEST")
        o.warmup_phase()
        for _ in range(6):
            o.step_update()
        path = Path(tempfile.mkdtemp(prefix="reward_trials_ckpt_")) / "u00006.pt"
        o.save(path)
        restored = TR.TrialOrchestratorV1.restore(path, collector_factory=factory, seed=17, authorization_sha256="TEST")
        self.assertEqual(restored.state_digest(), o.state_digest())
        for _ in range(3):
            a, b = o.step_update(), restored.step_update()
            self.assertEqual((a.critic_loss, a.actor_loss, a.reward_mean), (b.critic_loss, b.actor_loss, b.reward_mean))
        self.assertEqual(restored.state_digest(), o.state_digest())
        with self.assertRaises(TR.TrialTrainingError):                         # other arm's binding refuses the bundle
            TR.TrialOrchestratorV1.restore(path, collector_factory=lambda: collector(17, "C1"), seed=17,
                                           authorization_sha256="TEST")

    def test_authorization_binds_arm_reward_source_and_evidence(self):
        auth = Path(tempfile.mkdtemp(prefix="reward_trials_auth_")) / "AUTH.json"
        A.issue(auth, binding_dir=BINDING, authorized_by="TEST", authorized_on="TEST", scope="unit tests only")
        A.verify(auth, arm="C2", seed=43, binding_dir=BINDING)
        with self.assertRaises(A.AuthorizationError):
            A.verify(auth, arm="C3", seed=43, binding_dir=BINDING)
        with self.assertRaises(A.AuthorizationError):
            A.verify(auth, arm="C1", seed=5, binding_dir=BINDING)
        with self.assertRaises(A.AuthorizationError):
            A.issue(auth, binding_dir=BINDING, authorized_by="TEST", authorized_on="TEST", scope="again")


if __name__ == "__main__":
    unittest.main()

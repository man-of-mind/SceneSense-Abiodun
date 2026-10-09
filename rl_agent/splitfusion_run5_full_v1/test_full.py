"""Focused CPU checks for the full-support Run-5 environment and actor binding: action support, state/SNR, latency
accounting, reward, checkpoint restoration.

Run from abiodun/:  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 -m unittest -v \
    rl_agent.splitfusion_run5_full_v1.test_full
"""
import collections
import json
import math
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
from rl_agent.splitfusion_corrected_latency_env_v1.outcome import classify
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import ConditionalHybridActor
from rl_agent.splitfusion_run5_restricted_pilot_v1.quality import candidate_from_row
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import authorization as A, collector as C, latency as L, quality as Q, support as S, trainer as TR

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "experiments/splitfusion_run5_full_v1/binding_01"
torch.set_num_threads(4)


class Fixture:
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            shared = C.load_shared_sources()
            cls._cache = dict(shared=shared, lat=L.FullSupportLatencyModelV1(BINDING, shared["model"]))
        return cls._cache


def collector(seed=17):
    f = Fixture.get()
    return C.FullRun5CollectorV1(latency=f["lat"], seed=seed, shared_sources=f["shared"])


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


class SupportTest(unittest.TestCase):
    def test_actor_is_the_original_class_with_original_bounds_and_no_mask(self):
        o = TR.FullOrchestratorV1(collector_factory=collector, seed=17, authorization_sha256="TEST")
        self.assertIs(type(o.actor), ConditionalHybridActor)
        self.assertEqual(S.BOUNDS, ((8812, 9800), (8493, 9800), (7491, 9800), (8280, 9800), (7827, 9800), (5902, 9800),
                                    (6603, 9800), (5682, 9800), (1946, 9800), (3110, 9800), (1217, 9800), (0, 9791)))
        states = torch.tensor([d["state"] for _, d in random_decisions(collector(), 40)], dtype=torch.float32)
        with torch.no_grad():
            sample = o.actor.sample_all_modes(states, generator=torch.Generator().manual_seed(0))
        self.assertTrue(bool((sample.probs > 0).all()), "every mode must have positive probability (no mask)")
        q = sample.q_e4
        for m, (lo, hi) in enumerate(S.BOUNDS):
            self.assertTrue(bool(((q[:, m] >= lo) & (q[:, m] <= hi)).all()))
        self.assertFalse(any("mask" in name for name, _ in o.actor.named_buffers()))

    def test_warmup_is_the_original_12_mode_schedule(self):
        schedule = RT.warmup_schedule(17)
        actions = [schedule.action_at(i) for i in range(len(schedule))]
        self.assertEqual(len(actions), 288)
        cells = collections.Counter((a.mode_id, a.q_bin_index) for a in actions)
        self.assertEqual(set(cells), {(m, b) for m in range(12) for b in range(6)})
        self.assertTrue(all(v == 4 for v in cells.values()))
        self.assertTrue(all(S.in_support(a.mode_id, a.q_e4) for a in actions))

    def test_out_of_support_action_is_refused_and_payload_stays_in_transport_support(self):
        col = collector()
        bad = orch.ModeledActionRequestV1(decision_ordinal=0, mode_id=0, q_e4=5000, source="STOCHASTIC_ACTOR",
                                          warmup_q_bin_index=None)
        with self.assertRaises(C.FullCollectorError):
            col.collect_transition(bad)
        cat = Fixture.get()["shared"]["catalog"]
        support = Fixture.get()["shared"]["model"].fit_support
        for m, (lo, hi) in enumerate(S.BOUNDS):
            wires = [cat.draw(k, mode_id=m, q_e4=q)[2] for k in cat.keys for q in (lo, hi)]
            self.assertGreaterEqual(min(wires), support["min_wire_bytes"])
            self.assertLessEqual(max(wires), support["max_wire_bytes"])


class StateTest(unittest.TestCase):
    def test_22d_state_causal_snr_mcs_and_previous_q_perc(self):
        rows = random_decisions(collector(29), 650)          # 1,300 ticks > one balanced 4 x 300-tick block
        profiles, prev_checked = set(), 0
        for i, (tr, d) in enumerate(rows):
            state = d["state"]
            self.assertEqual(len(state), 22)
            self.assertAlmostEqual(state[21], min(1.0, max(0.0, (d["snr_db"] - 5.5) / 19.0)), places=6)
            self.assertAlmostEqual(state[2], d["prior_ul_mcs"] / 28.0, places=6)     # pre-action MCS in state
            profiles.add(d["profile"])
            prev = rows[i - 1][1] if i else None
            if prev and not prev["excluded"] and not d["excluded"] and prev["terminal"] == "SUCCESS" and tr is not None:
                self.assertEqual(state[4 + prev["mode_id"]], 1.0)
                self.assertAlmostEqual(state[17], prev["q_perc"], places=12)            # Q_perc, not the reward value
                self.assertEqual(state[20], 1.0)
                prev_checked += 1
            if prev and not prev["excluded"] and prev["terminal"] == "TIMEOUT" and not d["excluded"]:
                self.assertEqual((state[17], state[18], state[20]), (0.0, 0.0, 0.0))
        self.assertGreater(prev_checked, 20)
        self.assertEqual(profiles, {"FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"})

    def test_next_state_snr_is_not_generated_before_the_action(self):
        col = collector(43)
        before = col.current_state_features()
        channel_before = json.dumps(col._channel.checkpoint(), sort_keys=True, default=str)
        random_decisions(col, 1)
        self.assertNotEqual(channel_before, json.dumps(col._channel.checkpoint(), sort_keys=True, default=str))
        self.assertEqual(col.diagnostics[0]["state"], before)


class LatencyTest(unittest.TestCase):
    def test_zero_delta_reproduces_every_measured_base_chain_exactly(self):
        lat = Fixture.get()["lat"]
        for cid, chain in lat.chains.items():
            ctx = chain.context
            ranked = lat.ranked(ctx.camera_si, ctx.radar_p40)
            d = lat.move(cid, ranked=ranked, mode_id=ctx.mode_id, q_e4=ctx.q_e4, prior_ul_mcs=ctx.prior_ul_mcs,
                         backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes, u=1.0)
            self.assertEqual(d.delta_ms["C"], 0.0)
            self.assertEqual(d.delta_ms["F"], 0.0)
            self.assertAlmostEqual(d.delta_ms["E"], 0.0, places=12)
            self.assertFalse(d.uplink_late)
            composed = compose(d.vector, carry_in=chain.service.carry_in)
            self.assertAlmostEqual(composed.receipt, chain.instants["ue_receipt"], places=6, msg=cid)

    def test_every_stage_once_gt_concurrent_and_receipt_bounds(self):
        rows = random_decisions(collector(17), 300)
        for _, d in rows:
            stages = d["stage_ms"]
            pred = sum(stages[s] for s in "ABCDEFH")
            self.assertAlmostEqual(pred, d["prediction_at_service_ms"], places=6)
            self.assertGreaterEqual(d["receipt_ms"], max(pred, d["gt_ready_ms"]))       # GT concurrent, not added
            self.assertLess(d["receipt_ms"], pred + d["gt_ready_ms"])
        lat = Fixture.get()["lat"]
        cid = sorted(lat.chains)[0]
        ctx = lat.chains[cid].context
        d = lat.move(cid, ranked=lat.ranked(ctx.camera_si, ctx.radar_p40), mode_id=2, q_e4=9000, prior_ul_mcs=20,
                     backlog_bytes=0.0, wire_bytes=150_000, u=1.0)
        path = compose(d.vector).critical_path
        labels = [label for label, _ in path]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertAlmostEqual(sum(v for _, v in path), compose(d.vector).receipt, places=6)

    def test_uplink_responds_to_payload_mcs_and_lateness_forces_timeout(self):
        lat = Fixture.get()["lat"]
        cid = sorted(lat.chains)[3]
        ctx = lat.chains[cid].context
        ranked = lat.ranked(ctx.camera_si, ctx.radar_p40)
        e = lambda wire, mcs, backlog=0.0: lat.move(cid, ranked=ranked, mode_id=11, q_e4=5000, prior_ul_mcs=mcs,
                                                    backlog_bytes=backlog, wire_bytes=wire, u=1.0).vector.E
        self.assertGreater(e(400_000, 26), e(50_000, 26))
        self.assertGreater(e(200_000, 10), e(200_000, 26))
        late = lat.move(cid, ranked=ranked, mode_id=11, q_e4=5000, prior_ul_mcs=9, backlog_bytes=3_000_000,
                        wire_bytes=400_000, u=0.0)
        self.assertTrue(late.uplink_late)
        receipt = compose(late.vector).receipt
        self.assertGreater(receipt, 170.0)
        self.assertEqual(classify(int(round(receipt * 1e6)), 0.5).kind, contract.RewardEventKind.TIMEOUT)

    def test_inclusive_deadline_and_common_random_numbers(self):
        self.assertEqual(classify(170_000_000, 0.4).kind, contract.RewardEventKind.DELIVERED_SUCCESS)
        self.assertEqual(classify(170_000_001, 0.4).kind, contract.RewardEventKind.TIMEOUT)
        lat = Fixture.get()["lat"]
        r1, r2 = random.Random(9), random.Random(9)
        lat.draw(mode_id=0, q_e4=9800, camera_si=10.0, radar_p40=0.2, prior_ul_mcs=20, backlog_bytes=0.0,
                 wire_bytes=70_000, rng=r1)
        lat.draw(mode_id=11, q_e4=500, camera_si=10.0, radar_p40=0.2, prior_ul_mcs=12, backlog_bytes=5_000.0,
                 wire_bytes=200_000, rng=r2)
        self.assertEqual(r1.getstate(), r2.getstate())


class RewardTest(unittest.TestCase):
    def test_reward_value_equals_the_agreed_definition_on_grid_rows(self):
        cat = Fixture.get()["shared"]["catalog"]
        con = sqlite3.connect(f"file:{ROOT / V1.PAYLOAD_AUTHORITY_RELPATH}?mode=ro&immutable=1", uri=True)
        rows = [json.loads(b) for (b,) in con.execute(
            "SELECT row_json FROM quality_rows WHERE grid_split = ? LIMIT 400", (SS.FIT_SPLIT,))]
        con.close()
        checked = 0
        for row in rows:
            key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
            if key not in set(cat.keys):
                continue
            got = cat.reward_value(key, mode_id=int(row["mode_id"]), q_e4=int(row["q_e4"]))
            want = candidate_from_row(row)
            if want is None:
                self.assertIsNone(got)
            else:
                num = den = 0.0
                for cls, tol, w in (("person", 1.2, 0.6), ("vehicle", 1.0, 0.4)):
                    if row[f"loc_{cls}_eligible_gt"]:
                        num += w * sum(e <= tol for e in row[f"loc_{cls}_matched_xy_errors_m"]) / row[f"loc_{cls}_eligible_gt"]
                        den += w
                seg = 1.0 if row["q_seg"] is None else 0.7 + 0.3 * row["q_seg"]
                self.assertAlmostEqual(got, num / den * seg, places=12)
            checked += 1
        self.assertGreater(checked, 100)

    def test_outcome_rewards_and_session_reset(self):
        col = collector(17)
        rows = random_decisions(col, 1500)
        terminals = collections.Counter(d["terminal"] for _, d in rows)
        self.assertGreater(terminals["EVALUATOR_FAULT"], 0)
        sessions = 1
        for tr, d in rows:
            if d["terminal"] == "SUCCESS":
                self.assertEqual(tr.reward, d["reward_value"])
                self.assertLessEqual(d["receipt_ms"], 170.0)
                self.assertIsNotNone(d["q_perc"])
            elif d["terminal"] == "TIMEOUT":
                self.assertEqual(tr.reward, 0.0)
                self.assertTrue(d["receipt_ms"] is None or d["receipt_ms"] > 170.0)
            else:
                self.assertIsNone(tr)
                self.assertTrue(d["excluded"] and d["q_perc"] is None and d["receipt_ms"] <= 170.0)
                sessions += 1
        self.assertEqual(col._session_index + 1, sessions)
        late_undefined = [d for _, d in rows if d["terminal"] == "TIMEOUT" and d["q_perc"] is None]
        self.assertTrue(all(not d["excluded"] for d in late_undefined))


class CheckpointTest(unittest.TestCase):
    def test_bundle_restore_is_digest_equal_and_continues_bit_identically(self):
        o = TR.FullOrchestratorV1(collector_factory=collector, seed=17, authorization_sha256="TEST")
        o.warmup_phase()
        for _ in range(6):
            o.step_update()
        path = Path(tempfile.mkdtemp(prefix="full_run5_ckpt_")) / "u00006.pt"
        o.save(path)
        restored = TR.FullOrchestratorV1.restore(path, collector_factory=collector, seed=17, authorization_sha256="TEST")
        self.assertEqual(restored.state_digest(), o.state_digest())
        for _ in range(3):
            a, b = o.step_update(), restored.step_update()
            self.assertEqual((a.critic_loss, a.actor_loss, a.reward_mean), (b.critic_loss, b.actor_loss, b.reward_mean))
        self.assertEqual(restored.state_digest(), o.state_digest())
        with self.assertRaises(TR.FullTrainingError):
            o.save(path)                                                                # create-only

    def test_authorization_binds_source_and_evidence(self):
        auth = Path(tempfile.mkdtemp(prefix="full_run5_auth_")) / "AUTH.json"
        A.issue(auth, binding_dir=BINDING, authorized_by="TEST", authorized_on="TEST", scope="unit tests only")
        A.verify(auth, seed=43, binding_dir=BINDING)
        with self.assertRaises(A.AuthorizationError):
            A.verify(auth, seed=5, binding_dir=BINDING)
        with self.assertRaises(A.AuthorizationError):
            A.issue(auth, binding_dir=BINDING, authorized_by="TEST", authorized_on="TEST", scope="again")


if __name__ == "__main__":
    unittest.main()

"""Tests for the versioned backlog transition (CPU)."""
from __future__ import annotations

import json
import math
import random
import unittest
from pathlib import Path

from .transition import RateTableV1, TransitionError, next_backlog

R = Path(__file__).resolve().parents[2] / "experiments/splitfusion_run5_backlog_transition_v1/20261007_v1"
T = RateTableV1(((0, 10), (11, 15), (16, 28)), (4000.0, 5000.0, 10000.0), 20.0)


class TransitionUnit(unittest.TestCase):
    def test_carry_only_drains_linearly_without_access_delay(self):
        b = next_backlog(carry_bytes=50_000, events=[], t_sample=4.0, mcs_k=9, mcs_k1=9, t_switch=math.inf, table=T)
        self.assertAlmostEqual(b, 50_000 - 4 * 4000)

    def test_enqueue_on_empty_waits_access_delay(self):
        b = next_backlog(carry_bytes=0, events=[(10.0, 100_000)], t_sample=30.0, mcs_k=20, mcs_k1=20, t_switch=math.inf, table=T)
        self.assertEqual(b, 100_000)                       # served only from 30 ms
        b = next_backlog(carry_bytes=0, events=[(10.0, 100_000)], t_sample=35.0, mcs_k=20, mcs_k1=20, t_switch=math.inf, table=T)
        self.assertAlmostEqual(b, 100_000 - 5 * 10_000)

    def test_overlapping_frames_are_served_once(self):
        """Two frames on a busy queue are not drained independently: same result as one combined enqueue."""
        a = next_backlog(carry_bytes=10_000, events=[(1.0, 40_000), (1.0, 60_000)], t_sample=9.0, mcs_k=9, mcs_k1=9, t_switch=math.inf, table=T)
        b = next_backlog(carry_bytes=10_000, events=[(1.0, 100_000)], t_sample=9.0, mcs_k=9, mcs_k1=9, t_switch=math.inf, table=T)
        self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(a, 110_000 - 9 * 4000)

    def test_byte_conservation_random(self):
        rng = random.Random(7)
        for _ in range(2000):
            carry = rng.choice([0.0, rng.uniform(0, 2e5)])
            ev = [(rng.uniform(0, 200), rng.uniform(0, 2e5)) for _ in range(rng.randint(0, 3))]
            ts = rng.uniform(1, 300)
            sw = rng.uniform(0, 300)
            m0, m1 = rng.randint(8, 28), rng.randint(8, 28)
            b = next_backlog(carry_bytes=carry, events=ev, t_sample=ts, mcs_k=m0, mcs_k1=m1, t_switch=sw, table=T)
            entered = carry + sum(x for t, x in ev if t < ts)
            self.assertGreaterEqual(b, 0.0)
            self.assertLessEqual(b, entered + 1e-6)
            self.assertGreaterEqual(b, entered - 10_000 * ts - 1e-6)      # cannot serve faster than the max rate

    def test_rate_switch_uses_successor_mcs(self):
        b = next_backlog(carry_bytes=100_000, events=[], t_sample=10.0, mcs_k=9, mcs_k1=20, t_switch=5.0, table=T)
        self.assertAlmostEqual(b, 100_000 - 5 * 4000 - 5 * 10_000)

    def test_refusals(self):
        with self.assertRaises(TransitionError):
            next_backlog(carry_bytes=-1, events=[], t_sample=1, mcs_k=9, mcs_k1=9, t_switch=1, table=T)
        with self.assertRaises(TransitionError):
            next_backlog(carry_bytes=0, events=[(-1.0, 5.0)], t_sample=1, mcs_k=9, mcs_k1=9, t_switch=1, table=T)


class LiveCounterexample(unittest.TestCase):
    def test_fade_299_to_300_reproduced_within_factor_two(self):
        rows = [json.loads(l) for l in (R / "validation_01/PER_TRANSITION.jsonl").read_text().splitlines()]
        r = next(x for x in rows if x["session"] == "fade" and x["k"] == 299)
        self.assertEqual(r["obs"], 117734.0)
        self.assertGreater(r["pred"], 1000.0)
        self.assertLess(abs(math.log(r["pred"] / r["obs"])), math.log(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RecordedEvidence(unittest.TestCase):
    def test_contract_reproduced_exactly(self):
        c = json.loads((R / "contract_01/CONTRACT_CHECK.json").read_text())
        for name, s in c["sessions"].items():
            self.assertEqual(s["A_selector"]["bytes_match"], s["A_selector"]["n"], name)
            self.assertEqual(s["A_selector"]["age_match_1ms"], s["A_selector"]["n"], name)

    def test_disabled_reproduces_old_environment(self):
        e = json.loads((R / "env_check_01/ENV_CHECK.json").read_text())
        self.assertTrue(e["identity"]["identical"])
        self.assertGreaterEqual(e["identity"]["n"], 600)

    def test_registered_acceptance(self):
        v = json.loads((R / "validation_01/VALIDATION.json").read_text())
        self.assertEqual(v["acceptance"]["decision"], "READY_FOR_SEPARATELY_APPROVED_TRAINING_COMPARISON")

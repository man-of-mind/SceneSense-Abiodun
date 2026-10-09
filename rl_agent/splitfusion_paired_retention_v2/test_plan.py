"""Pre-launch sampling plan: determinism, digest binding, full-route sampling and outcome independence."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from . import fixtures as F, observer as OB, plan as P


def plan_config(**kwargs):
    plan = P.build_plan(plan_id="test", **kwargs)
    return P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan))


class PlanTests(unittest.TestCase):
    def test_1000_frame_plan_is_100_evenly_spaced_ordinals(self):
        plan = P.build_plan(plan_id="run5_valid", planned_frames=1000)
        self.assertEqual((plan["planned_policy_decisions"], plan["stride"], plan["offset"]), (500, 5, 2))
        self.assertEqual(plan["ordinals"], list(range(2, 500, 5)))
        self.assertEqual(len(plan["ordinals"]), 100)
        self.assertEqual(plan["bounds"]["max_host_bytes"], P.HOST_BYTES_CEILING)
        self.assertLessEqual(100 * P.INPUT_BYTES, plan["bounds"]["max_host_bytes"])
        self.assertEqual(P.plan_sha256(plan), P.plan_sha256(P.build_plan(plan_id="run5_valid", planned_frames=1000)))

    def test_write_load_and_tamper(self):
        plan = P.build_plan(plan_id="run5_valid", planned_frames=1000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PLAN.json"
            digest = P.write_plan(path, plan)
            self.assertEqual(P.load_plan(path, digest), plan)
            config = P.RetentionConfigV2.from_plan_file(path, digest)
            self.assertTrue(config.enabled)
            with self.assertRaises(FileExistsError):
                P.write_plan(path, plan)
            with self.assertRaisesRegex(P.PlanError, "digest"):
                P.load_plan(path, "0" * 64)
            forged = dict(plan, ordinals=[0] + plan["ordinals"][1:])     # hand-picked ordinal
            forged_path = Path(tmp) / "FORGED.json"
            forged_path.write_bytes(P.canonical(forged))
            with self.assertRaisesRegex(P.PlanError, "registered rule"):
                P.load_plan(forged_path, P.sha256_bytes(P.canonical(forged)))
        for bad in (dict(max_host_bytes=P.HOST_BYTES_CEILING + 1), dict(max_storage_bytes=P.STORAGE_BYTES_CEILING + 1),
                    dict(target_records=P.MAX_RECORDS_CEILING + 1), dict(persistence="SYNC_DISK")):
            with self.subTest(bad=bad), self.assertRaises(P.PlanError):
                P.build_plan(plan_id="x", planned_frames=1000, **bad)

    def test_disabled_by_default(self):
        config = P.RetentionConfigV2()
        self.assertFalse(config.enabled)
        observer = OB.make_observer()
        self.assertIsInstance(observer, OB.NoOpRetentionObserver)
        tensor = F.shared_input()
        planned, prepared = F.make_decision(frame_id=10, tensor_seq=5, decision_seq=0,
                                            session_uuid="00000000-0000-0000-0000-000000000001")
        self.assertIsNone(observer.begin(planned))
        self.assertIs(observer.observe_input(tensor, "cuda:0"), tensor)
        self.assertIsNone(observer.end(None, prepared))
        with OB.install_retention(observer) as info:
            self.assertEqual(info, {"enabled": False})
        with self.assertRaises(P.PlanError):
            P.RetentionConfigV2(enabled=True)                       # no plan -> refused
        with self.assertRaises(P.PlanError):
            P.RetentionConfigV2(enabled=False, plan=P.build_plan(plan_id="x", planned_frames=100))

    def run_route(self, frames=999, seed=7, planned_frames=1000):
        observer = OB.make_observer(plan_config(planned_frames=planned_frames))
        events = list(F.route(frames, seed=seed))
        F.drive(observer, events, F.shared_input())
        return observer, events

    def test_full_route_retains_about_100_evenly_spaced_decisions_without_raising(self):
        observer, events = self.run_route()
        decisions = [e for e in events if e[0] == "decision"]
        observed = observer.counters["policy_decisions_observed"]
        self.assertEqual(observed, len(decisions))
        self.assertTrue(480 <= observed <= 500, observed)
        held = sorted(o for o, s in observer.status.items() if s == "HELD")
        expected = [o for o in observer.plan["ordinals"] if o < observed]
        self.assertEqual(held, expected)
        self.assertTrue(95 <= len(held) <= 100)
        self.assertEqual({b - a for a, b in zip(held, held[1:])}, {5})
        self.assertEqual(observer.counters["faults"], 0)
        self.assertEqual(observer.counters["budget_exhausted"], 0)
        self.assertEqual(observer.counters["skipped_off_sample"] + observer.counters["sampled"], observed)
        self.assertGreater(observer.counters["non_decision_frames"], 400)   # holds + fallbacks never counted
        # The retained decisions are the planned ordinals of the policy-decision sequence, not the first 100.
        frames = [decisions[o][1].frame_id for o in held]
        self.assertEqual(frames, [observer.records[o]["frame_id"] for o in held])
        self.assertGreater(frames[-1] - frames[0], 1500)

    def test_sample_independent_of_latency_quality_action_and_payload(self):
        base, _ = self.run_route(seed=7)
        rng = random.Random(99)
        observer = OB.make_observer(plan_config(planned_frames=1000))
        events = list(F.route(999, seed=7))
        shuffled = []
        for kind, planned, prepared, outcome in events:     # same decision sequence, new outcomes/actions
            if kind == "decision":
                outcome = {k: rng.random() for k in outcome}
                planned2, prepared2 = F.make_decision(frame_id=planned.frame_id, tensor_seq=planned.tensor_seq,
                    decision_seq=planned.decision_seq, session_uuid=planned.session_uuid,
                    q_e4=rng.randint(3000, 8000), mode_id=rng.randint(0, 11), payload=rng.randbytes(rng.randint(1, 4096)))
                shuffled.append((kind, planned2, prepared2, outcome))
            else:
                shuffled.append((kind, planned, prepared, outcome))
        # Production failures on off-sample decisions must not shift the sample either.
        off = [i for i in range(500) if i not in set(base.plan["ordinals"])]
        F.drive(observer, shuffled, F.shared_input(), fail=lambda i: i in off[::7])
        self.assertEqual({o for o, s in observer.status.items() if s == "HELD"},
                         {o for o, s in base.status.items() if s == "HELD"})
        self.assertEqual({r["frame_id"] for r in observer.records.values()},
                         {r["frame_id"] for r in base.records.values()})

    def test_shorter_and_longer_runs_are_well_defined(self):
        with tempfile.TemporaryDirectory() as tmp:
            short = OB.make_observer(plan_config(planned_frames=1000))
            F.drive(short, list(F.route(400, seed=3)), F.shared_input())
            observed = short.counters["policy_decisions_observed"]
            not_reached = [o for o in short.plan["ordinals"] if o >= observed]
            self.assertTrue(not_reached)
            long = OB.make_observer(plan_config(planned_frames=400))
            F.drive(long, list(F.route(999, seed=3)), F.shared_input())
            self.assertGreater(long.counters["skipped_beyond_plan"], 250)
            self.assertEqual(sum(s == "HELD" for s in long.status.values()), 100)
            self.assertEqual(long.counters["budget_exhausted"], 0)
            self.assertEqual(long.counters["faults"], 0)
            del tmp


if __name__ == "__main__":
    unittest.main()

"""CPU unit tests for BoundedGcPolicy (in-process; no worker, no corpus)."""
import gc
import socket
import unittest
import weakref

from .gc_policy import BoundedGcPolicy


class Node:
    pass


def make_cycles(n):
    for _ in range(n):
        a, b = Node(), Node()
        a.b, b.a = b, a


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.state = (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count())

    def tearDown(self):
        gc.unfreeze()
        gc.set_threshold(*self.state[1])
        (gc.enable if self.state[0] else gc.disable)()

    def test_install_uninstall_restores_state(self):
        for enabled in (True, False):
            (gc.enable if enabled else gc.disable)()
            gc.set_threshold(700, 10, 10)
            p = BoundedGcPolicy(hard_cap_gen1=150).install()
            self.assertTrue(gc.isenabled())
            self.assertEqual(gc.get_threshold(), (700, 10, 150))
            with self.assertRaises(RuntimeError):
                p.install()
            for k in range(3):
                p.after_reply("EVIDENCE_DURABLE")
            self.assertGreater(gc.get_freeze_count(), 0)
            p.uninstall()
            self.assertEqual((gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()), (enabled, (700, 10, 10), 0))
            p.uninstall()   # idempotent

    def test_refuses_foreign_frozen_generation(self):
        gc.freeze()
        with self.assertRaises(RuntimeError):
            BoundedGcPolicy().install()
        gc.unfreeze()

    def test_invalid_bounds(self):
        for kw in (dict(warmup_decisions=0), dict(hard_cap_gen1=5), dict(idle_min_gen1=-1)):
            with self.assertRaises(ValueError):
                BoundedGcPolicy(**kw)

    def test_hard_cap_forces_full_collection_without_idle_maintenance(self):
        gc.set_threshold(700, 10, 10)
        p = BoundedGcPolicy(hard_cap_gen1=12).install()
        gc.collect()
        full = []
        cb = lambda phase, info: full.append(info) if phase == "stop" and info["generation"] == 2 else None
        gc.callbacks.append(cb)
        try:
            victim = Node(); victim.self = victim
            ref = weakref.ref(victim)
            del victim
            # survive young generations so the cycle reaches gen2 before it can be reclaimed
            gc.collect(1)
            # Long-lived survivors (as the worker's retained state) make gen2 growth pending; CPython's full
            # collection needs gen1 count > threshold2 AND pending > long_lived_total/4.
            keep = []
            for _ in range(701 * 11 * 13 * 3):   # ~3x the allocations needed for hard_cap gen1 collections
                keep.append(Node())
            self.assertTrue(full, "no automatic full collection despite the hard cap")
            self.assertIsNone(ref(), "gen2 cyclic garbage not reclaimed under the hard cap")
            self.assertLessEqual(max(0, gc.get_count()[2]), 12)
        finally:
            gc.callbacks.remove(cb)
            p.uninstall()

    def test_idle_maintenance_collects_and_skips_when_request_pending(self):
        p = BoundedGcPolicy(warmup_decisions=1).install()
        try:
            self.assertIsNone(p.after_reply("FREEZE"))                     # never on critical-path ops
            r = p.after_reply("EVIDENCE_DURABLE")
            self.assertEqual(r["kind"], "warmup_collect_freeze")
            make_cycles(700 * 10 * 2)
            ref_holder = Node(); ref_holder.self = ref_holder
            ref = weakref.ref(ref_holder); del ref_holder
            r = p.after_reply("EVIDENCE_DURABLE")
            self.assertEqual(r["kind"], "idle_full_collect")
            self.assertIsNone(ref())
            self.assertEqual(gc.get_count()[2], 0)
            a, b = socket.socketpair()
            try:
                b.send(b"x")
                make_cycles(700 * 10 * 2)
                self.assertIsNone(p.after_reply("EVIDENCE_DURABLE", a))
                self.assertEqual(p.events[-1]["kind"], "skipped_pending")
            finally:
                a.close(); b.close()
        finally:
            p.uninstall()


if __name__ == "__main__":
    unittest.main()

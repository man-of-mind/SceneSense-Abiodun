"""Focused exactness/lifecycle tests for the bounded PairHistory summaries."""
from dataclasses import dataclass
import gc
import random
from types import SimpleNamespace
import weakref

import numpy as np
import unittest

from rl_agent.splitfusion_gt_storage_accounting_v1 import test_accounting as V1T
from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as CF
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS

from . import accounting as A, switch

OLD = OWN.payload_storage_bytes
PH = CF._PairHistory


def conflict_snapshot(history, templates=None, decisions=None):
    """Small real-shape producer snapshot used by cache exactness regressions."""
    return {"templates": {} if templates is None else templates,
            "pairs": {("a", "b"): history},
            "decisions": {} if decisions is None else decisions,
            "frame_count": 1, "row_count": 1, "previous_frame": 1,
            "previous_timestamp": 1., "first_timestamp": 1., "episode": "e",
            "seen_samples": {"s"}, "chain": "c"}


def outcome(function, *values):
    try:
        return ("ok", function(*values))
    except BaseException as error:
        return ("raise", type(error), str(error))


def assert_same(session, *values):
    old, new = outcome(OLD, *values), outcome(session, *values)
    assert old == new, (old, new)
    return old


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.session = A.AccountingSession(max_entries=64)

    def tearDown(self):
        self.session.close()

    def test_aliases_and_shared_best(self):
        p1 = PH().appended(.8, .1, .2, "s1")
        p2 = p1.appended(.1, .3, .4, "s2")
        self.assertIs(p1.best, p2.best)
        for values in ((p1, p2), (p2, p1), ({"a": p1, "b": p2}, p1.best),
                       ([p1, p1, p2],), ((p2, p1), p2.best)):
            assert_same(self.session, *values)
            assert_same(self.session, *reversed(values))
        self.assertGreater(self.session.snapshot()["alias_fallbacks"], 0)

    def test_randomized_alias_differential(self):
        rng = random.Random(20261009)
        for _ in range(1200):
            pool = []
            values = [V1T._random_structure(rng, pool) for _ in range(rng.randrange(1, 4))]
            assert_same(self.session, *values)
            assert_same(self.session, *reversed(values))

    def test_unsupported_errors_unchanged(self):
        cases = ([object()], np.array([object()], dtype=object), ["\ud800"], [lambda: 0])
        for value in cases:
            assert_same(self.session, value)

    def test_real_snapshot_commit_cancel_rollback(self):
        state = SimpleNamespace(templates={}, pairs={}, decisions={}, frame_count=0, row_count=0,
                                previous_frame=-1, previous_timestamp=-1., first_timestamp=None,
                                episode=None, seen_samples=set(), chain="c0")
        producer = SimpleNamespace(conflict_state=state, tracker=SimpleNamespace(_ages={}, _last_time={}))
        for frame in range(30):
            restore = {"ages": dict(producer.tracker._ages), "last": dict(producer.tracker._last_time),
                       "conflict": NS.NumericSourceProducer._conflict_snapshot(producer)}
            key = ("a", "b")
            state.pairs[key] = state.pairs.get(key, PH()).appended(.2, float(frame), .1, f"s{frame}")
            staged = {"conflict": dict(pairs=state.pairs), "_restore": restore}
            assert_same(self.session, b"plan", staged, None)
            assert_same(self.session, b"plan", staged, (state.pairs[key],))
            if frame % 2:
                NS.NumericSourceProducer._restore_history(producer, restore)
                assert_same(self.session, {"_restore": {"ages": {}, "last": {},
                            "conflict": NS.NumericSourceProducer._conflict_snapshot(producer)}})
            else:
                staged["_restore"] = {}
                assert_same(self.session, b"plan", staged, None)

    def test_mutation_invalidates(self):
        history = PH().appended(.2, .3, .4, "s")
        assert_same(self.session, history)
        before = self.session.snapshot()["invalidations"]
        object.__setattr__(history, "overlap_count", history.overlap_count + 1)
        assert_same(self.session, history)
        self.assertGreater(self.session.snapshot()["invalidations"], before)

    def test_snapshot_nested_mutation_invalidates_and_aliases(self):
        history = PH().appended(.2, .3, .4, "s")
        state = SimpleNamespace(templates={("a", "b"): {"name": "x", "v": 1.}},
                                pairs={("a", "b"): history},
                                decisions={("a", "b"): {"name": "x", "v": 1.}},
                                frame_count=1, row_count=2, previous_frame=3,
                                previous_timestamp=1., first_timestamp=1., episode="e",
                                seen_samples={"s"}, chain="c")
        snapshot = NS.NumericSourceProducer._conflict_snapshot(SimpleNamespace(conflict_state=state))
        assert_same(self.session, snapshot)
        assert_same(self.session, snapshot)
        self.assertGreater(self.session.snapshot()["snapshot_hits"], 0)
        before = self.session.snapshot()["snapshot_invalidations"]
        snapshot["decisions"][("a", "b")]["v"] = 2.
        assert_same(self.session, snapshot)
        self.assertGreater(self.session.snapshot()["snapshot_invalidations"], before)
        assert_same(self.session, snapshot, snapshot)

    def test_snapshot_context_never_poisoned_as_full_summary(self):
        history = PH().appended(.2, .3, .4, "s")
        snapshot = conflict_snapshot(history)
        self.assertEqual(OLD(history.iou, snapshot), 752)
        self.assertEqual(self.session(history.iou, snapshot), 752)
        self.assertEqual(self.session(snapshot), 752)
        self.assertEqual(self.session.snapshot()["snapshots"], 2)

        reverse = A.AccountingSession(max_entries=16)
        try:
            self.assertEqual(reverse(snapshot), 752)
            self.assertEqual(reverse(history.iou, snapshot), 752)
            self.assertEqual(reverse(snapshot), 752)
            self.assertGreater(reverse.snapshot()["snapshot_alias_fallbacks"], 0)
        finally:
            reverse.close()

    def test_snapshot_changed_argument_order_and_shared_descendants(self):
        history = PH().appended(.2, .3, .4, "s")
        snapshot = conflict_snapshot(history)
        sequences = ((snapshot, history.iou), (history.iou, snapshot),
                     (history.best, snapshot, history.iou),
                     (snapshot, history.best, history.iou))
        for values in sequences:
            assert_same(self.session, *values)
            assert_same(self.session, snapshot)

    def test_snapshot_representation_changing_nested_mutations(self):
        history = PH().appended(.2, .3, .4, "s")
        row = {"v": 1}
        snapshot = conflict_snapshot(history, templates={("a", "b"): row})
        self.assertEqual(OLD(snapshot), 809)
        self.assertEqual(self.session(snapshot), 809)

        row["v"] = np.array(1, dtype=np.int64)
        self.assertEqual(OLD(snapshot), 833)
        self.assertEqual(self.session(snapshot), 833)

        row["v"] = np.array(1, dtype=np.int32)
        self.assertEqual(OLD(snapshot), 829)
        self.assertEqual(self.session(snapshot), 829)

        row["v"] = 1
        self.assertEqual(self.session(snapshot), OLD(snapshot))
        object.__setattr__(history, "iou", (np.array(1, dtype=np.int64),))
        assert_same(self.session, snapshot)

    def test_owner_stage_shortcut_revalidates_snapshot_structure(self):
        history = PH().appended(.2, .3, .4, "s")
        row_value = {"v": 1}
        snapshot = conflict_snapshot(history, templates={("a", "b"): row_value})
        staged = {"_restore": {"conflict": snapshot}}
        products = ()
        row = SimpleNamespace(plan=None, staged=staged, products=products, ready=None,
                              reservation=10_000, payload_bytes=None)
        owner = SimpleNamespace(_failure=None)
        owner._fail = lambda message, exc_type=OWN.OwnershipError: OWN.ProducerOwner._fail(
            owner, message, exc_type)

        def stage_products():
            OWN.ProducerOwner._account(owner, row)

        def bind_plan():
            OWN.ProducerOwner._account(owner, row)

        def set_ready():
            OWN.ProducerOwner._account(owner, row)

        session = switch.install(max_entries=16)
        try:
            stage_products()
            self.assertEqual(row.payload_bytes, OLD(row.plan, staged, products))
            row_value["v"] = np.array(1, dtype=np.int64)
            row.plan = b"p"
            bind_plan()
            self.assertEqual(row.payload_bytes, OLD(row.plan, staged, products))
            row_value["v"] = np.array(1, dtype=np.int32)
            set_ready()
            self.assertEqual(row.payload_bytes, OLD(row.plan, staged, products))
            self.assertEqual(session.snapshot()["owner_shortcut_hits"], 0)
        finally:
            switch.uninstall()

    def test_stale_identity_guard(self):
        old = PH().appended(.1, .2, .3, "old")
        new = PH().appended(.4, .5, .6, "new")
        signature = A._signature(old)
        logical, descendants = A._summary(old)
        self.session._entries[id(new)] = A._Entry(weakref.ref(old), signature, descendants, logical)
        before = self.session.snapshot()["invalidations"]
        assert_same(self.session, new)
        self.assertGreater(self.session.snapshot()["invalidations"], before)

    def test_missing_dict_mutation_tag_fails_closed(self):
        history = PH().appended(.2, .3, .4, "s")
        snapshot = conflict_snapshot(history)
        saved = A._dict_version
        A._dict_version = None
        try:
            assert_same(self.session, snapshot)
            assert_same(self.session, snapshot)
            self.assertEqual(self.session.snapshot()["snapshots"], 0)
        finally:
            A._dict_version = saved

    def test_weak_cleanup_bound_and_close(self):
        values = [PH().appended(.1, float(i), .2, str(i)) for i in range(200)]
        for value in values:
            assert_same(self.session, value)
        self.assertLessEqual(self.session.snapshot()["entries"], 64)
        self.assertGreater(self.session.snapshot()["evictions"], 0)
        values.clear(); del value
        gc.collect()
        self.assertLessEqual(self.session.snapshot()["entries"], 1)
        self.session.close()
        self.assertEqual(self.session.snapshot()["entries"], 0)

    def test_capacity_exact_and_one_byte_under(self):
        staged = {"pairs": {("a", "b"): PH().appended(.5, .1, .1, "s")},
                  "rows": [{"x": 1.}] * 3, "img": np.zeros(10)}
        exact = OLD(b"plan", staged, None)
        for delta, succeeds in ((0, True), (-1, False)):
            switch.install(max_entries=16)
            try:
                row = SimpleNamespace(plan=b"plan", staged=staged, products=None, ready=None,
                                      reservation=exact + delta, payload_bytes=None)
                owner = SimpleNamespace(_failure=None)
                owner._fail = lambda message, exc_type=OWN.OwnershipError: OWN.ProducerOwner._fail(owner, message, exc_type)
                result = outcome(OWN.ProducerOwner._account, owner, row)
                self.assertEqual(result[0] == "ok", succeeds)
                self.assertEqual(row.payload_bytes, exact)
            finally:
                switch.uninstall()

    def test_switch_off_by_default_reversible(self):
        self.assertFalse(switch.installed())
        session = switch.install(max_entries=8)
        try:
            self.assertTrue(switch.installed())
            self.assertIs(OWN.payload_storage_bytes, session)
        finally:
            switch.uninstall()
        self.assertIs(OWN.payload_storage_bytes, switch.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        self.assertTrue(session.snapshot()["closed"])


if __name__ == "__main__":
    unittest.main()

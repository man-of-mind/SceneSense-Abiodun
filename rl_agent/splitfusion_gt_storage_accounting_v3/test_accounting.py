"""Focused exactness/lifecycle tests: V3 accounting vs the ORIGINAL ownership.payload_storage_bytes (CPU only).

The oracle is ``switch.ORIGINAL_PAYLOAD_STORAGE_BYTES``, verified by identity to be the function defined in
ownership.py (never an installed session).  Carries forward the V1 and V2 suites (incl. the four V2 counterexamples:
752/712 context poisoning in both orders, 809/833/829 representation change, 938/914 owner-stage mutation) and adds
V3-specific tests.

  cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \\
      /usr/bin/python3 -B -m unittest -v rl_agent.splitfusion_gt_storage_accounting_v3.test_accounting
"""
from dataclasses import dataclass, field
import gc
import random
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import unittest

from rl_agent.splitfusion_gt_storage_accounting_v1 import test_accounting as V1T
from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as CF
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS

from . import accounting as A, switch

OLD = switch.ORIGINAL_PAYLOAD_STORAGE_BYTES
PH = CF._PairHistory
assert switch.is_original(OLD) and not isinstance(OLD, A.AccountingSession)


def outcome(function, *values):
    try:
        return ("ok", function(*values))
    except BaseException as error:
        return ("raise", type(error), str(error))


def same(session, *values):
    old, new = outcome(OLD, *values), outcome(session, *values)
    assert old == new, (old, new)
    return old


def stateless(*values):
    return same(A.payload_storage_bytes, *values)


def conflict_snapshot(history, templates=None, decisions=None):
    """Small real-shape producer snapshot (the V2 counterexample fixture)."""
    return {"templates": {} if templates is None else templates,
            "pairs": {("a", "b"): history},
            "decisions": {} if decisions is None else decisions,
            "frame_count": 1, "row_count": 1, "previous_frame": 1,
            "previous_timestamp": 1., "first_timestamp": 1., "episode": "e",
            "seen_samples": {"s"}, "chain": "c"}


def producer_state(templates=None):
    state = SimpleNamespace(templates=dict(templates or {}), pairs={}, decisions=dict(templates or {}),
                            frame_count=0, row_count=0, previous_frame=-1, previous_timestamp=-1.,
                            first_timestamp=None, episode=None, seen_samples=set(), chain="c0")
    return SimpleNamespace(conflict_state=state, tracker=SimpleNamespace(_ages={}, _last_time={}))


def snapshot_of(producer):
    return NS.NumericSourceProducer._conflict_snapshot(producer)


def advance(producer, rng, frame, keys):
    """One derive-like step: new PairHistory versions, new/changed decision rows, new template rows, a new sample."""
    s = producer.conflict_state
    updates = {}
    for key in rng.sample(keys, rng.randrange(1, min(5, len(keys) + 1))):
        updates[key] = s.pairs.get(key, PH()).appended(rng.random(), rng.random(), rng.random(), f"s{frame}é")
    s.pairs.update(updates)
    for key in updates:
        if key not in s.templates:
            s.templates[key] = {"name": f"n{key[0]}", "geom": rng.random(), "flag": bool(frame % 2)}
        s.decisions[key] = dict(s.templates[key], count=len(s.pairs[key].iou), adjudication="REJECTED_DISTINCT")
    s.seen_samples.add(f"s{frame}é")
    s.frame_count += 1
    s.chain = f"c{frame}"
    return updates


# ------------------------------------------------------------------ owner stage helpers (caller = function name)
def _owner():
    owner = SimpleNamespace(_failure=None)
    owner._fail = lambda message, exc_type=OWN.OwnershipError: OWN.ProducerOwner._fail(owner, message, exc_type)
    return owner


def compute_unbound(owner, row):
    OWN.ProducerOwner._account(owner, row)


def stage_products(owner, row):
    OWN.ProducerOwner._account(owner, row)


def bind_plan(owner, row):
    OWN.ProducerOwner._account(owner, row)


def set_ready(owner, row):
    OWN.ProducerOwner._account(owner, row)


STAGES = dict(compute_unbound=compute_unbound, stage_products=stage_products, bind_plan=bind_plan,
              set_ready=set_ready)


def expected(row):
    value = OLD(row.plan, row.staged, row.products)
    if row.ready is not None:
        value += len(row.ready.metadata_bytes) + len(row.ready.segmentation_bytes) + len(row.ready.object_ignore_bytes)
    return value


def run_stage(stage, owner, row):
    want = outcome(expected, row)
    got = outcome(STAGES[stage], owner, row)
    if want[0] == "ok":
        assert got == ("ok", None), got
        assert row.payload_bytes == want[1], (stage, row.payload_bytes, want[1])
    else:
        assert got == want, (got, want)
    return want


def new_row(staged, plan=None, products=None, reservation=10 ** 12):
    return SimpleNamespace(plan=plan, staged=staged, products=products, ready=None, reservation=reservation,
                           payload_bytes=None)


READY = SimpleNamespace(metadata_bytes=b"m" * 7, segmentation_bytes=b"s" * 11, object_ignore_bytes=b"o" * 3)


@dataclass
class Mutable:
    a: object = None
    b: object = None


@dataclass(frozen=True)
class Frozen:
    a: object = None
    b: tuple = ()


@dataclass(frozen=True)
class OtherHistory:  # same field names as _PairHistory, different class
    iou: tuple = ()
    ranges: tuple = ()
    near: tuple = ()
    overlap_count: int = 0
    best: tuple = (-1., "")


class SubHistory(PH):
    pass


# ------------------------------------------------------------------ randomized snapshot structures
def random_scalar(rng, rare=True):
    pool = [0.5, 1, True, None, "ascii", "ünï", "日本語", np.float64(1.5), np.float32(2), np.int64(7), np.uint8(3),
            np.bool_(True), np.int32(-1), "", float("nan")]
    if rare and rng.random() < 0.02:
        return rng.choice([np.array(1), [1.0], b"raw", bytearray(b"ab"), "\ud800", object()])
    return rng.choice(pool)


def random_snapshot(rng, size=12, rare=True):
    names = [f"o{i}" for i in range(size)]
    keys = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    shared_rows = [{"k": random_scalar(rng, rare), "v": rng.random()} for _ in range(4)]
    templates, decisions, pairs = {}, {}, {}
    for key in rng.sample(keys, min(len(keys), rng.randrange(0, 20))):
        row = rng.choice(shared_rows) if rng.random() < 0.3 else {"a": random_scalar(rng, rare), "b": rng.random()}
        templates[key] = row
        if rng.random() < 0.6:
            decisions[key] = row if rng.random() < 0.5 else dict(row, d=random_scalar(rng, rare))
    histories = []
    for key in rng.sample(keys, min(len(keys), rng.randrange(0, 15))):
        if histories and rng.random() < 0.2:
            history = rng.choice(histories)                       # shared PairHistory object
        else:
            history = PH()
            for _ in range(rng.randrange(0, 5)):
                history = history.appended(rng.random(), rng.random(), rng.random(), random_scalar(rng, False)
                                           if rng.random() < 0.1 else "s")
        histories.append(history)
        pairs[tuple(list(key)) if rng.random() < 0.5 else key] = history       # equal but distinct key object
    samples = {rng.choice(["s", "ü", "日本"]) + str(i) for i in range(rng.randrange(0, 6))}
    if rare and rng.random() < 0.02:
        samples.add("\udfff")
    return {"templates": templates, "pairs": pairs, "decisions": decisions, "frame_count": rng.randrange(9),
            "row_count": rng.randrange(9), "previous_frame": rng.randrange(9),
            "previous_timestamp": rng.random(), "first_timestamp": rng.choice([None, 1.0]),
            "episode": rng.choice([None, "e", "épisode"]), "seen_samples": samples, "chain": "c"}


def members(snapshot):
    """Containers reachable from a snapshot (for prior-context and mutation targeting)."""
    out = [snapshot["templates"], snapshot["pairs"], snapshot["decisions"], snapshot["seen_samples"]]
    out += list(snapshot["templates"].values()) + list(snapshot["decisions"].values())
    out += list(snapshot["templates"]) + list(snapshot["pairs"])
    for history in snapshot["pairs"].values():
        out += [history] + [getattr(history, name) for name in ("iou", "ranges", "near", "best")
                            if not isinstance(getattr(history, name, 1), (int, float, str))]
    return out


def mutate_snapshot(rng, snapshot):
    choice = rng.randrange(10)
    rows = list(snapshot["templates"].values()) + list(snapshot["decisions"].values())
    histories = [value for value in snapshot["pairs"].values() if isinstance(value, (PH, OtherHistory))]
    if choice == 0 and rows:
        rng.choice(rows)[rng.choice(["a", "b", "new"])] = random_scalar(rng)
    elif choice == 1 and rows:
        row = rng.choice(rows)
        if row:
            key = next(iter(row)); value = row.pop(key); row[key] = value            # delete + reinsert same
    elif choice == 2 and histories:
        object.__setattr__(rng.choice(histories), "overlap_count",
                           rng.choice([3, np.int64(3), None, "x", 3.0]))
    elif choice == 3 and histories:
        object.__setattr__(rng.choice(histories), rng.choice(["iou", "ranges", "near", "best"]),
                           rng.choice([(1.0, 2.0), (), [1.0], (np.array(1),), (1.0, "x"), ("b",)]))
    elif choice == 4 and snapshot["pairs"]:
        key = rng.choice(list(snapshot["pairs"]))
        snapshot["pairs"][key] = rng.choice([PH(), PH().appended(.1, .2, .3, "z"), OtherHistory(), {"x": 1.0}])
    elif choice == 5:
        snapshot["seen_samples"].add(rng.choice(["new", "ü", 1.0, ("t",)]))
    elif choice == 6:
        snapshot["frame_count"] = rng.choice([1, 2.0, None, "x", np.float32(1), [1]])
    elif choice == 7 and snapshot["templates"]:
        key = rng.choice(list(snapshot["templates"]))
        snapshot["decisions"][key] = snapshot["templates"][key]                     # new alias
    elif choice == 8 and histories:
        history = rng.choice(histories)
        object.__setattr__(history, "__class__", rng.choice([SubHistory, PH]))
    else:
        snapshot["templates"][("z", str(rng.random()))] = {"q": rng.random()}


# ------------------------------------------------------------------ randomized whole-staged structures
def random_staged(rng, snapshot):
    rows = list(snapshot["decisions"].values())
    shared = [1.5, "x", np.float32(2)]
    evidence = {"rows": rows[:3] if rows else [], "values": [rng.random() for _ in range(4)]}
    backing = {"p": 1.0, "q": "ü"}
    captured = evidence
    def closure():
        return captured                          # a real closure cell (as Staged.same_class)
    staged = dict(frame={"frame_id": 3, "name": "ü"}, boxes=[{"x": rng.random(), "id": 1}, shared],
                  images=np.zeros((2, 3)), view=None, proxy=MappingProxyType(backing),
                  obj=Mutable(shared, (1.0, 2.0)), fn=closure, conflict=evidence,
                  _restore={"ages": {1: 2.0}, "last": {}, "conflict": snapshot}, tail=set(["a", 1.0]))
    staged["view"] = np.ones((3, 3))[:1]          # a view of an array that is never resized
    products = (Frozen(evidence, (shared,)), [np.arange(3.0)], {"m": rows[:1]})
    return staged, products, backing


def mutate_staged(rng, staged, products, backing):
    choice = rng.randrange(13)
    if choice == 0:
        staged["boxes"][0]["x"] = rng.choice([1, np.array(1), 2.5, "s"])
    elif choice == 1:
        staged["boxes"][1] = rng.choice([[1.5], "xy", 1.5, np.array(1)])
    elif choice == 2:
        staged["boxes"].append(rng.choice([1.0, {"y": 1}]))
    elif choice == 3:
        staged["images"].resize((rng.randrange(1, 4), 3), refcheck=False)
    elif choice == 4:
        object.__setattr__(products[0], "b", rng.choice([(), (1.0,), (np.array(1),)]))
    elif choice == 5:
        staged["fn"].__closure__[0].cell_contents = rng.choice([{"z": 1.0}, staged["conflict"], [1, 2]])
    elif choice == 6:
        staged["tail"].add(rng.choice(["b", 2.0, ("t",)]))
    elif choice == 7:
        backing[rng.choice(["p", "r"])] = rng.choice([1.0, "long string", np.array(1)])
    elif choice == 8:
        staged["obj"].a = rng.choice([1.0, "s", [1], None])
    elif choice == 9:
        products[1][0] = rng.choice([np.arange(5.0), np.arange(3.0), 1.0])
    elif choice == 10:
        staged["frame"]["name"] = rng.choice(["ü", "x" * 9, None])
    elif choice == 11:
        staged["extra"] = rng.choice([bytearray(b"abc"), b"abc", 1.0])
    else:
        mutate_snapshot(rng, staged["_restore"]["conflict"])


class V1CarriedTests(unittest.TestCase):
    """The V1 suite, run against the V3 stateless API and a persistent V3 session."""

    def setUp(self):
        self.session = A.AccountingSession()

    def tearDown(self):
        self.session.close()

    def both(self, *values):
        stateless(*values)
        return same(self.session, *values)

    def test_shared_child_tuples(self):
        t, s, d = (1.0, 2.0, 3.0), ("a", "bb"), {"x": 1.0, "y": "z"}
        for values in ([t, t], [(t, t)], [{"k": t, "j": t}], [[t, s, d], (s, d, t)], [d, [d, d]],
                       [Frozen(t, t), t]):
            self.both(*values)
            self.both(*reversed(values))

    def test_shared_empty(self):
        e = ()
        self.both(e, e, (), [(), ()], {"a": ()}, PH(), PH(), [], set(), frozenset(), {})
        self.both([PH(), PH()], (e,))

    def test_pair_history_best(self):
        p1 = PH().appended(0.8, 0.1, 0.2, "s1")
        p2 = p1.appended(0.1, 0.3, 0.4, "s2")
        p3 = p2.appended(0.9, 0.2, 0.1, "s3")
        self.assertIs(p2.best, p1.best)
        for values in ([p1, p2], [p2, p1], [{"k": p1}, {"k": p2}], [p1, p2, p3], [(p3, p2, p1)], [p2.best, p1, p2]):
            self.both(*values)
            self.both(*reversed(values))

    def test_top_level(self):
        t = (1.0,) * 40
        staged = {"a": t, "b": [t, {"c": t}], "s": {"x", "y"}}
        for args in ((b"plan", staged, None), (None, staged, staged), (staged, b"plan"), (staged["a"], staged)):
            self.both(*args)

    def test_randomized(self):
        rng = random.Random(20261008)
        for _ in range(3000):
            pool = []
            values = [V1T._random_structure(rng, pool) for _ in range(rng.randrange(1, 4))]
            self.both(*values)
            self.both(*reversed(values))

    def test_subclasses(self):
        self.both(V1T.ListSub([1.0, 2.0]), [V1T.StrSub("ab"), "cd"], (V1T.FloatSub(1.0), 2.0),
                  V1T.DictSub(a=1.0), {"a": V1T.StrSub("q")}, {V1T.StrSub("k"): 1}, [np.str_("np"), "s"],
                  (np.bytes_(b"b"),), [np.complex64(1)], [np.longdouble(1)])

    def test_rejections(self):
        cases = [[1.0, np.array([object()], dtype=object)], {"a": 1.0, "b": object()}, (1.0, object()),
                 ["ok", "\ud800"], {"\udfff": 1.0}, {"\udfff", "a"}, ("a", "b\ud800c"), [1, 2, lambda: 0], [Ellipsis]]
        for value in cases:
            self.both(value)
            self.both(b"x", value)

    def test_closures(self):
        a = np.arange(12.0).reshape(3, 4)
        masks = {"car": a[:1], "person": a}
        def closure():
            return masks, a
        self.both(closure, masks, a, a[1:])
        self.both(Mutable(closure, (a, a.T)))

    def test_rollback(self):
        rng = random.Random(7)
        producer = producer_state({("a", "b"): {"k": 1.0}})
        keys = [(f"o{i}", f"o{j}") for i in range(6) for j in range(i + 1, 6)]
        for frame in range(60):
            restore = {"ages": dict(producer.tracker._ages), "last": dict(producer.tracker._last_time),
                       "conflict": snapshot_of(producer)}
            before = self.both({"_restore": restore})
            updates = advance(producer, rng, frame, keys)
            staged = {"conflict": {"q": [list(k) for k in updates]}, "_restore": restore}
            self.both(b"plan", staged, None)
            self.both(b"plan", staged, {"products": tuple(updates.values())})
            if frame % 3 == 0:
                NS.NumericSourceProducer._restore_history(producer, restore)
                self.assertEqual(self.both({"_restore": {"ages": {}, "last": {},
                                                         "conflict": snapshot_of(producer)}}), before)
            else:
                staged["_restore"] = {}
                self.both(b"plan", staged, None)

    def test_stateless_api_keeps_no_module_state(self):
        names = {name: id(value) for name, value in vars(A).items() if not name.startswith("__")}
        t = tuple(float(i) for i in range(100))
        stateless(t, {"a": t}, conflict_snapshot(PH().appended(.1, .2, .3, "s")))
        self.assertEqual({name: id(value) for name, value in vars(A).items() if not name.startswith("__")}, names)


class V2CarriedTests(unittest.TestCase):
    """The V2 suite incl. its four counterexamples, adapted to V3 counters where V2 internals were asserted."""

    def setUp(self):
        self.session = A.AccountingSession(max_objects=64)

    def tearDown(self):
        self.session.close()

    def test_aliases_and_shared_best(self):
        p1 = PH().appended(.8, .1, .2, "s1")
        p2 = p1.appended(.1, .3, .4, "s2")
        self.assertIs(p1.best, p2.best)
        for values in ((p1, p2), (p2, p1), ({"a": p1, "b": p2}, p1.best), ([p1, p1, p2],), ((p2, p1), p2.best)):
            same(self.session, *values)
            same(self.session, *reversed(values))

    def test_randomized_alias_differential(self):
        rng = random.Random(20261009)
        for _ in range(1200):
            pool = []
            values = [V1T._random_structure(rng, pool) for _ in range(rng.randrange(1, 4))]
            same(self.session, *values)
            same(self.session, *reversed(values))

    def test_unsupported_errors_unchanged(self):
        for value in ([object()], np.array([object()], dtype=object), ["\ud800"], [lambda: 0]):
            same(self.session, value)

    def test_real_snapshot_commit_cancel_rollback(self):
        producer = producer_state()
        state = producer.conflict_state
        for frame in range(30):
            restore = {"ages": dict(producer.tracker._ages), "last": dict(producer.tracker._last_time),
                       "conflict": snapshot_of(producer)}
            key = ("a", "b")
            state.pairs[key] = state.pairs.get(key, PH()).appended(.2, float(frame), .1, f"s{frame}")
            staged = {"conflict": dict(pairs=state.pairs), "_restore": restore}
            same(self.session, b"plan", staged, None)
            same(self.session, b"plan", staged, (state.pairs[key],))
            if frame % 2:
                NS.NumericSourceProducer._restore_history(producer, restore)
                same(self.session, {"_restore": {"ages": {}, "last": {}, "conflict": snapshot_of(producer)}})
            else:
                staged["_restore"] = {}
                same(self.session, b"plan", staged, None)

    def test_mutation_invalidates(self):
        history = PH().appended(.2, .3, .4, "s")
        same(self.session, history)
        object.__setattr__(history, "overlap_count", history.overlap_count + 1)
        same(self.session, history)
        snapshot = conflict_snapshot(history)
        same(self.session, snapshot)
        object.__setattr__(history, "overlap_count", np.int64(5))
        same(self.session, snapshot)

    def test_snapshot_nested_mutation_invalidates_and_aliases(self):
        history = PH().appended(.2, .3, .4, "s")
        state = SimpleNamespace(templates={("a", "b"): {"name": "x", "v": 1.}}, pairs={("a", "b"): history},
                                decisions={("a", "b"): {"name": "x", "v": 1.}}, frame_count=1, row_count=2,
                                previous_frame=3, previous_timestamp=1., first_timestamp=1., episode="e",
                                seen_samples={"s"}, chain="c")
        snapshot = snapshot_of(SimpleNamespace(conflict_state=state))
        same(self.session, snapshot)
        same(self.session, snapshot)
        snapshot["decisions"][("a", "b")]["v"] = 2.
        same(self.session, snapshot)
        same(self.session, snapshot, snapshot)
        self.assertGreater(self.session.snapshot()["row_misses"], 0)

    def test_snapshot_context_never_poisoned_as_full_summary(self):
        """V2 counterexample 1/2: 752 vs 712 (context-partial result reused as full), both argument orders."""
        history = PH().appended(.2, .3, .4, "s")
        snapshot = conflict_snapshot(history)
        self.assertEqual(OLD(history.iou, snapshot), 752)
        self.assertEqual(self.session(history.iou, snapshot), 752)
        self.assertEqual(self.session(snapshot), 752)
        reverse = A.AccountingSession(max_objects=16)
        try:
            self.assertEqual(reverse(snapshot), 752)
            self.assertEqual(reverse(history.iou, snapshot), 752)
            self.assertEqual(reverse(snapshot), 752)
            self.assertEqual(reverse(snapshot, history.iou), OLD(snapshot, history.iou))
        finally:
            reverse.close()

    def test_snapshot_changed_argument_order_and_shared_descendants(self):
        history = PH().appended(.2, .3, .4, "s")
        snapshot = conflict_snapshot(history)
        for values in ((snapshot, history.iou), (history.iou, snapshot), (history.best, snapshot, history.iou),
                       (snapshot, history.best, history.iou)):
            same(self.session, *values)
            same(self.session, snapshot)

    def test_snapshot_representation_changing_nested_mutations(self):
        """V2 counterexample 3: 809 -> 833 (int -> int64 ndarray) -> 829 (int32 ndarray)."""
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
        same(self.session, snapshot)

    def test_owner_stage_shortcut_revalidates_snapshot_structure(self):
        """V2 counterexample 4: owner-stage reuse after nested mutation (938 vs 914)."""
        history = PH().appended(.2, .3, .4, "s")
        row_value = {"v": 1}
        snapshot = conflict_snapshot(history, templates={("a", "b"): row_value})
        staged = {"_restore": {"conflict": snapshot}}
        row, owner = new_row(staged, products=(), reservation=10_000), _owner()
        session = switch.install(max_objects=16)
        try:
            run_stage("stage_products", owner, row)
            row_value["v"] = np.array(1, dtype=np.int64)
            row.plan = b"p"
            run_stage("bind_plan", owner, row)
            row_value["v"] = np.array(1, dtype=np.int32)
            run_stage("set_ready", owner, row)
            self.assertEqual(session.snapshot()["owner_shortcut_hits"], 0)
            self.assertEqual(session.snapshot()["owner_shortcut_rejections"], 2)
        finally:
            switch.uninstall()

    def test_identity_reuse_after_pin_release(self):
        rng = random.Random(3)
        for _ in range(200):
            history = PH().appended(rng.random(), .3, .4, "s")
            snapshot = conflict_snapshot(history, templates={(str(rng.random()), "b"): {"v": 1.0}})
            same(self.session, snapshot)
            del history, snapshot
            gc.collect()
        self.assertLessEqual(self.session.memo_objects(), 64)

    def test_missing_dict_mutation_tag_fails_closed(self):
        saved_tag, saved_flag = A._dict_version, A.FAST_PATH_AVAILABLE
        try:
            A._dict_version = None
            self.assertIsNotNone(A.capabilities()["dict_tags"])
            A.FAST_PATH_AVAILABLE = False
            session = A.AccountingSession()
            self.assertFalse(session.fast_path)
            snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
            same(session, snapshot)
            same(session, snapshot)
            self.assertEqual(session.snapshot()["snapshot_fast"], 0)
            self.assertEqual(session.memo_objects(), 0)
        finally:
            A._dict_version, A.FAST_PATH_AVAILABLE = saved_tag, saved_flag

    def test_bound_and_close(self):
        histories = [PH().appended(.1, float(i), .2, str(i)) for i in range(200)]
        snapshot = conflict_snapshot(histories[0])
        snapshot["pairs"] = {(str(i), "x"): history for i, history in enumerate(histories)}
        same(self.session, snapshot)                                  # 200 pairs > 64 objects: exact fallback
        self.assertEqual(self.session.snapshot()["snapshot_bound_fallbacks"], 1)
        self.assertEqual(self.session.memo_objects(), 0)
        same(self.session, conflict_snapshot(histories[1]))
        self.assertGreater(self.session.memo_objects(), 0)
        self.session.close()
        self.assertEqual(self.session.memo_objects(), 0)
        self.assertEqual(len(self.session._tuple_pins), 0)
        with self.assertRaises(RuntimeError):
            self.session(1)

    def test_capacity_exact_and_one_byte_under(self):
        snapshot = conflict_snapshot(PH().appended(.5, .1, .1, "s"))
        staged = {"pairs": {("a", "b"): PH().appended(.5, .1, .1, "s")}, "rows": [{"x": 1.}] * 3,
                  "img": np.zeros(10), "_restore": {"conflict": snapshot}}
        exact = OLD(b"plan", staged, None)
        for delta, succeeds in ((0, True), (-1, False)):
            for stage in ("compute_unbound", "stage_products"):
                switch.install(max_objects=16)
                try:
                    row, owner = new_row(staged, b"plan", reservation=exact + delta), _owner()
                    result = outcome(STAGES[stage], owner, row)
                    self.assertEqual(result[0] == "ok", succeeds)
                    if not succeeds:
                        self.assertIs(result[1], OWN.CapacityError)
                    self.assertEqual(row.payload_bytes, exact)
                finally:
                    switch.uninstall()

    def test_switch_off_by_default_reversible(self):
        self.assertFalse(switch.installed())
        self.assertIs(OWN.payload_storage_bytes, OLD)
        session = switch.install(max_objects=8)
        try:
            self.assertTrue(switch.installed())
            self.assertIs(OWN.payload_storage_bytes, session)
            with self.assertRaises(RuntimeError):
                switch.install()
        finally:
            switch.uninstall()
        self.assertIs(OWN.payload_storage_bytes, switch.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        self.assertTrue(session.snapshot()["closed"])
        OWN.payload_storage_bytes = lambda *values: 0
        try:
            with self.assertRaises(RuntimeError):
                switch.install()
        finally:
            switch.uninstall()
            OWN.payload_storage_bytes = OLD


class V3SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.session = A.AccountingSession()

    def tearDown(self):
        self.session.close()

    def test_capability_in_worker_interpreter(self):
        self.assertEqual(sys.executable, "/usr/bin/python3")
        self.assertTrue(A.FAST_PATH_AVAILABLE, A.capabilities())
        self.assertTrue(all(reason is None for reason in A.capabilities().values()))
        self.assertTrue(self.session.fast_path)

    def test_fast_path_is_actually_used(self):
        producer = producer_state({("a", "b"): {"k": 1.0}})
        advance(producer, random.Random(1), 0, [("a", "b"), ("a", "c")])
        same(self.session, snapshot_of(producer))
        self.assertEqual(self.session.snapshot()["snapshot_fast"], 1)
        self.assertEqual(self.session.snapshot()["snapshot_fallbacks"], 0)

    def test_shared_empty_best_tuples_and_rows(self):
        p0 = PH()
        p1 = p0.appended(.9, .1, .1, "s1")
        p2 = p1.appended(.1, .2, .2, "s2")                  # shares best with p1
        row = {"a": 1.0, "n": "x"}
        snapshot = conflict_snapshot(p0, templates={("a", "b"): row, ("c", "d"): row}, decisions={("a", "b"): row})
        snapshot["pairs"].update({("c", "d"): p1, ("e", "f"): p2, ("g", "h"): PH()})
        for values in ((snapshot,), (p1.best, snapshot), (snapshot, p1.best), ((), snapshot), (row, snapshot),
                       (snapshot, p2, p0), ([p2.iou, p2.ranges], snapshot)):
            same(self.session, *values)
            same(self.session, *reversed(values))
            stateless(*values)

    def test_frame_data_shares_rows_with_rollback_history(self):
        rng = random.Random(11)
        producer = producer_state({("a", "b"): {"k": 1.0}})
        keys = [(f"o{i}", f"o{j}") for i in range(5) for j in range(i + 1, 5)]
        for frame in range(40):
            snapshot = snapshot_of(producer)
            advance(producer, rng, frame, keys)
            shared_rows = list(snapshot["decisions"].values())[:rng.randrange(0, 4)]
            evidence = {"rows": shared_rows, "pairs": list(snapshot["pairs"].values())[:2]}
            for staged in ({"conflict": evidence, "_restore": {"conflict": snapshot}},
                           {"_restore": {"conflict": snapshot}, "conflict": evidence}):
                same(self.session, b"p", staged, None)
                same(self.session, None, staged, (evidence,))

    def test_snapshot_members_encountered_earlier_in_call(self):
        rng = random.Random(5)
        for _ in range(400):
            snapshot = random_snapshot(rng)
            pool = members(snapshot)
            prior = rng.sample(pool, min(len(pool), rng.randrange(0, 6)))
            same(self.session, *prior, snapshot)
            same(self.session, snapshot, *prior)
            same(self.session, {"x": prior}, [snapshot, prior])

    def test_scalar_kinds_and_utf8(self):
        row = {"f": np.float32(1.5), "i": np.int64(2), "b": True, "n": None, "s": "日本語", "e": "",
               "nb": np.bool_(False), "u": np.uint16(4), "h": np.float16(1)}
        history = PH(iou=(1.0, np.float64(2.0), 3), ranges=(np.float32(1.0),), near=(True,), overlap_count=np.int64(2),
                     best=(0.5, "ünï"))
        snapshot = conflict_snapshot(history, templates={("ü", "日"): row}, decisions={("ü", "日"): dict(row, s="x")})
        snapshot["seen_samples"] = {"s", "ü", "日本語", ""}
        snapshot["episode"] = "épisode"
        snapshot["first_timestamp"] = None
        same(self.session, snapshot)
        same(self.session, snapshot, snapshot["templates"])
        self.assertEqual(self.session.snapshot()["snapshot_fallbacks"], 0)

    def test_nested_edits_deletion_reinsertion_and_type_changes(self):
        history = PH().appended(.2, .3, .4, "s")
        row = {"v": 1.0, "w": "x"}
        snapshot = conflict_snapshot(history, templates={("a", "b"): row}, decisions={("a", "b"): row})
        same(self.session, snapshot)
        for value in (2, 2.0, "longer string", None, np.float32(1), True, np.array(1), [1.0], (1.0,), {"q": 1},
                      "\ud800", object(), 1.0):
            row["v"] = value
            same(self.session, snapshot)
            same(self.session, b"p", {"_restore": {"conflict": snapshot}}, None)
        del row["w"]; same(self.session, snapshot)
        row["w"] = "x"; same(self.session, snapshot)
        snapshot["decisions"].pop(("a", "b")); same(self.session, snapshot)
        snapshot["decisions"][("a", "b")] = row; same(self.session, snapshot)
        snapshot["seen_samples"].discard("s"); same(self.session, snapshot)
        snapshot["seen_samples"].add("s"); same(self.session, snapshot)

    def test_pair_history_field_rebinding_and_class_changes(self):
        history = PH().appended(.2, .3, .4, "s").appended(.5, .6, .7, "t")
        snapshot = conflict_snapshot(history)
        same(self.session, snapshot)
        for name, value in (("iou", (1.0, 2.0, 3.0)), ("ranges", ()), ("near", [1.0]), ("near", (1.0,)),
                            ("best", (1.0, "ü")), ("best", 1.0), ("overlap_count", None), ("overlap_count", "x"),
                            ("overlap_count", np.int64(9)), ("iou", (np.array(1),)), ("iou", ((1.0,),)),
                            ("overlap_count", 2), ("iou", history.ranges), ("best", history.iou)):
            object.__setattr__(history, name, value)
            same(self.session, snapshot)
            same(self.session, history.iou, snapshot)
        object.__delattr__(history, "near")                         # falls back to the class default ()
        same(self.session, snapshot)
        object.__setattr__(history, "__class__", SubHistory)
        same(self.session, snapshot)
        object.__setattr__(history, "__class__", PH)
        same(self.session, snapshot)
        snapshot["pairs"][("a", "b")] = OtherHistory()
        same(self.session, snapshot)
        snapshot["pairs"][("a", "b")] = {"iou": ()}
        same(self.session, snapshot)

    def test_unsupported_and_refusals_match_original(self):
        for mutation in (lambda s: s["templates"].__setitem__(("a", "b"), {"x": np.array([object()], dtype=object)}),
                         lambda s: s["templates"].__setitem__(("a", "b"), {"x": object()}),
                         lambda s: s["seen_samples"].add("\udfff"),
                         lambda s: s["templates"].__setitem__(("\ud800", "b"), {"x": 1.0}),
                         lambda s: s.__setitem__("chain", object()),
                         lambda s: s.__setitem__("episode", "\ud800"),
                         lambda s: s["pairs"].__setitem__(("a", "c"), object()),
                         lambda s: object.__setattr__(s["pairs"][("a", "b")], "iou", (object(),))):
            snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
            mutation(snapshot)
            same(self.session, snapshot)
            same(self.session, b"p", {"_restore": {"conflict": snapshot}, "z": [object()]}, None)
            stateless(snapshot)

    def test_dict_key_subclasses_and_snapshot_lookalikes(self):
        snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
        lookalike = {V1T.StrSub(k) if k == "chain" else k: v for k, v in snapshot.items()}
        same(self.session, lookalike)
        same(self.session, V1T.DictSub(snapshot))
        snapshot["templates"] = V1T.DictSub({("a", "b"): {"v": 1.0}})
        same(self.session, snapshot)
        snapshot["templates"] = {("a", "b"): V1T.DictSub(v=1.0)}
        same(self.session, snapshot)
        snapshot["templates"] = {("a", V1T.StrSub("b")): {"v": 1.0}}
        same(self.session, snapshot)

    def test_pair_history_class_guard(self):
        snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
        same(self.session, snapshot)
        saved = A._PH_CLASS_TAG
        try:
            A._PH_CLASS_TAG = -1                      # as if _PairHistory's class dict had been modified
            same(self.session, snapshot)
            self.assertEqual(self.session.snapshot()["fallback_reasons"].get("pair_history_class_changed"), 1)
        finally:
            A._PH_CLASS_TAG = saved
        same(self.session, snapshot)

    def test_memo_overflow_returns_exact_fallback(self):
        rng = random.Random(9)
        small = A.AccountingSession(max_objects=12)
        try:
            for _ in range(300):
                snapshot = random_snapshot(rng, rare=False)
                same(small, snapshot)
                same(small, b"p", {"_restore": {"conflict": snapshot}}, None)
                self.assertLessEqual(small.memo_objects(), 12)
            self.assertGreater(small.snapshot()["snapshot_bound_fallbacks"], 0)
            self.assertGreater(small.snapshot()["snapshot_fast"], 0)
        finally:
            small.close()

    def test_randomized_snapshot_differential_with_mutations(self):
        rng = random.Random(20261010)
        snapshots = [random_snapshot(rng) for _ in range(6)]
        for step in range(4000):
            snapshot = rng.choice(snapshots)
            if rng.random() < 0.3:
                mutate_snapshot(rng, snapshot)
            pool = members(snapshot)
            prior = rng.sample(pool, min(len(pool), rng.randrange(0, 3)))
            other = rng.choice(snapshots)
            arrangement = rng.randrange(4)
            values = ((snapshot,), (*prior, snapshot), (snapshot, *prior, other),
                      ({"c": prior, "_restore": {"conflict": snapshot}}, other))[arrangement]
            same(self.session, *values)
            if step % 25 == 24:          # unsupported mutations persist; refresh so both paths stay exercised
                snapshots[rng.randrange(len(snapshots))] = random_snapshot(rng)
        counters = self.session.snapshot()
        self.assertGreater(counters["snapshot_fast"], 1000)
        self.assertGreater(counters["snapshot_fallbacks"], 0)


class V3OwnerChainTests(unittest.TestCase):
    """Four-stage owner chains through the real ProducerOwner._account (caller names as in production)."""

    def tearDown(self):
        switch.uninstall()

    def chain(self, owner, row, mutate_before=None):
        for stage in ("compute_unbound", "stage_products", "bind_plan", "set_ready"):
            if mutate_before is not None:
                mutate_before(stage)
            if stage == "stage_products" and row.products is None:
                row.products = ()
            if stage == "bind_plan":
                row.plan = b'{"plan":1}'
            if stage == "set_ready":
                row.ready = READY
            run_stage(stage, owner, row)

    def test_unchanged_chain_uses_shortcut(self):
        session = switch.install()
        producer = producer_state({("a", "b"): {"k": 1.0}})
        advance(producer, random.Random(2), 0, [("a", "b")])
        staged, products, _backing = random_staged(random.Random(2), snapshot_of(producer))
        row = new_row(staged)
        for stage, plan, ready in (("compute_unbound", None, None), ("stage_products", None, None),
                                   ("bind_plan", b"plan", None), ("set_ready", b"plan", READY)):
            row.plan, row.ready = plan, ready
            if stage == "stage_products":
                row.products = products
            run_stage(stage, _owner(), row)
        counters = session.snapshot()
        self.assertEqual(counters["owner_shortcut_hits"], 2)
        self.assertEqual(counters["owner_shortcut_rejections"], 0)
        self.assertEqual(counters["witness_ineligible"], 0)

    def test_every_mutation_kind_between_stages_is_detected(self):
        rng = random.Random(77)
        session = switch.install()
        rejected = 0
        for trial in range(600):
            producer = producer_state({("a", "b"): {"k": 1.0}})
            advance(producer, rng, trial, [("a", "b"), ("a", "c"), ("b", "c")])
            staged, products, backing = random_staged(rng, snapshot_of(producer))
            row, owner = new_row(staged), _owner()
            before = session.snapshot()["owner_shortcut_rejections"]
            for stage in ("compute_unbound", "stage_products", "bind_plan", "set_ready"):
                if stage == "stage_products":
                    row.products = products
                if stage in ("bind_plan", "set_ready"):
                    row.plan = rng.choice([b"plan", b"longer plan", "plän", None, b"plan"])
                    if rng.random() < 0.8:
                        mutate_staged(rng, staged, products, backing)
                if stage == "set_ready":
                    row.ready = READY
                run_stage(stage, owner, row)
            rejected += session.snapshot()["owner_shortcut_rejections"] > before
        self.assertGreater(rejected, 100)
        self.assertGreater(session.snapshot()["owner_shortcut_hits"], 50)

    def test_each_witness_kind_detects_its_mutation(self):
        session = switch.install()
        for name in ("list_element_scalar_to_array", "list_element_longer_string", "ndarray_resize",
                     "pair_history_overlap", "set_member", "mappingproxy_backing", "closure_cell",
                     "dataclass_field", "row_value", "samples_member", "map_entry", "ndarray_object_dtype"):
            history = PH().appended(.2, .3, .4, "s")
            row = {"v": 1.0}
            snapshot = conflict_snapshot(history, templates={("a", "b"): row})
            backing, items, array, members_set = {"p": 1.0}, [1, "ab"], np.zeros(4), {"x", 1.0}
            captured = {"c": 1.0}
            def closure():
                return captured
            frozen = Frozen({"x": 1.0}, (1.0,))
            staged = {"items": items, "array": array, "set": members_set, "proxy": MappingProxyType(backing),
                      "fn": closure, "_restore": {"conflict": snapshot}}
            row_, owner = new_row(staged, products=(frozen,)), _owner()
            run_stage("stage_products", owner, row_)
            before = session.snapshot()["owner_shortcut_rejections"]
            if name == "list_element_scalar_to_array":
                items[0] = np.array(1)
            elif name == "list_element_longer_string":
                items[1] = "abcdef"
            elif name == "ndarray_resize":
                array.resize((9,), refcheck=False)
            elif name == "ndarray_object_dtype":                    # same identity and nbytes (32)
                array.__setstate__((1, (4,), np.dtype("O"), False, ["a", "b", "c", "d"]))
            elif name == "pair_history_overlap":
                object.__setattr__(history, "overlap_count", "many")
            elif name == "set_member":
                members_set.discard("x"); members_set.add("xyz")
            elif name == "mappingproxy_backing":
                backing["p"] = "a longer string value"
            elif name == "closure_cell":
                closure.__closure__[0].cell_contents = {"c": np.zeros(3)}
            elif name == "dataclass_field":
                object.__setattr__(frozen, "b", (1.0, 2.0, 3.0))
            elif name == "row_value":
                row["v"] = np.array(1)
            elif name == "samples_member":
                snapshot["seen_samples"].add("new sample")
            else:
                snapshot["templates"][("c", "d")] = {"w": 1.0}
            row_.plan = b"plan"
            run_stage("bind_plan", owner, row_)
            self.assertEqual(session.snapshot()["owner_shortcut_rejections"], before + 1, name)

    def _array_dtype_case(self, dtype, placement, mutate_before):
        """review_01 reproducer shape: an ndarray whose dtype is changed in place by the public
        ``ndarray.__setstate__`` with identical identity and nbytes, between owner stages."""
        session = switch.install()
        array = np.zeros(1, dtype=np.int64)
        holder = {"direct": {"array": array}, "nested": {"rows": [array, {"again": array}], "_restore": {}},
                  "products": {"other": 1.0}}[placement]
        products = (array,) if placement == "products" else ()
        row, owner = new_row(holder, products=products), _owner()
        run_stage("stage_products", owner, row)
        for stage in ("bind_plan", "set_ready"):
            if stage == mutate_before:
                state = ((1, (1,), np.dtype("O"), False, ["x"]) if dtype == "object" else
                         (1, (1,), np.dtype(dtype), False, np.zeros(1, dtype=dtype).tobytes()))
                array.__setstate__(state)
                self.assertEqual(array.nbytes, 8)
            row.plan = b"p"
            if stage == "set_ready":
                row.ready = READY
            want = run_stage(stage, owner, row)      # integer or refusal must equal the original
            if dtype == "object" and stage == mutate_before:
                self.assertEqual(want[:2], ("raise", OWN.OwnershipError))
                self.assertEqual(want[2], "object ndarray cannot be retained")
                return session.snapshot()
        return session.snapshot()

    def test_ndarray_same_nbytes_object_dtype_keeps_original_refusal(self):
        """Regression for experiments/splitfusion_gt_storage_accounting_v3_review_01: before the fix the owner
        shortcut returned 102 where the original raises OwnershipError('object ndarray cannot be retained')."""
        for placement in ("direct", "nested", "products"):
            for stage in ("bind_plan", "set_ready"):
                counters = self._array_dtype_case("object", placement, stage)
                self.assertEqual(counters["owner_shortcut_rejections"], 1, (placement, stage))
                self.assertEqual(counters["owner_shortcut_hits"], 0 if stage == "bind_plan" else 1)
                switch.uninstall()

    def test_ndarray_same_nbytes_numeric_dtype_control_keeps_shortcut(self):
        """The valid control of review_01: a same-size numeric dtype change does not change the original integer
        (size depends only on nbytes), so the shortcut stays exact and is still used."""
        for dtype in ("float64", "uint64", "complex64"):              # 8-byte itemsize: same nbytes
            for placement in ("direct", "nested", "products"):
                counters = self._array_dtype_case(dtype, placement, "bind_plan")
                self.assertEqual(counters["owner_shortcut_hits"], 2, (dtype, placement))
                self.assertEqual(counters["owner_shortcut_rejections"], 0, (dtype, placement))
                switch.uninstall()

    def test_staged_dataclass_attribute_rebinding(self):
        session = switch.install()
        snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
        staged = Mutable({"_restore": {"conflict": snapshot}}, [1.0])
        products = (Frozen({"x": 1.0}, ()),)
        row, owner = new_row(staged), _owner()
        run_stage("compute_unbound", owner, row)
        row.products = products
        run_stage("stage_products", owner, row)
        staged.b = [1.0, 2.0, 3.0]                       # rebinding like lazy staged.evidence/inputs
        row.plan = b"plan"
        run_stage("bind_plan", owner, row)
        object.__setattr__(products[0], "a", {"x": np.zeros(5)})
        row.ready = READY
        run_stage("set_ready", owner, row)
        self.assertEqual(session.snapshot()["owner_shortcut_rejections"], 2)

    def test_ineligible_graphs_never_shortcut(self):
        session = switch.install()
        snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
        for extra in (bytearray(b"abc"), memoryview(b"abc"), V1T.StrSub("x"), V1T.FloatSub(1.0), V1T.DictSub(a=1),
                      V1T.ListSub([1.0]), np.str_("s"), np.complex64(1)):
            staged = {"_restore": {"conflict": snapshot}, "extra": [extra]}
            row, owner = new_row(staged), _owner()
            for stage in ("compute_unbound", "stage_products", "bind_plan", "set_ready"):
                row.products = () if stage != "compute_unbound" else None
                row.plan = b"plan" if stage in ("bind_plan", "set_ready") else None
                if stage == "bind_plan" and type(extra) is bytearray:
                    extra.extend(b"longer")             # a per-reference scalar whose size changes in place
                run_stage(stage, owner, row)
        self.assertEqual(session.snapshot()["owner_shortcut_hits"], 0)
        self.assertGreater(session.snapshot()["witness_ineligible"], 0)

    def test_plan_and_identity_rules(self):
        session = switch.install()
        snapshot = conflict_snapshot(PH().appended(.2, .3, .4, "s"))
        staged = {"_restore": {"conflict": snapshot}}
        row, owner = new_row(staged, products=()), _owner()
        run_stage("stage_products", owner, row)
        row.plan = b"plan"
        run_stage("bind_plan", owner, row)               # the only shortcut of this test
        row.plan = bytearray(b"mutable plan")           # not an exact immutable plan: full call
        run_stage("set_ready", owner, row)
        row.plan = b"plan"
        run_stage("set_ready", owner, row)               # set_ready -> set_ready is not an owner chain
        row.products = (1.0,)                            # different products object: full call
        run_stage("set_ready", owner, row)
        run_stage("compute_unbound", owner, row)         # compute_unbound -> bind_plan is not an owner chain
        run_stage("bind_plan", owner, row)
        same(session, b"x", staged, row.products)        # foreign (non-owner) call resets the chain
        run_stage("set_ready", owner, row)
        self.assertEqual(session.snapshot()["owner_shortcut_hits"], 1)
        self.assertEqual(session.snapshot()["owner_shortcut_rejections"], 0)

    def test_commit_hold_unsent_rollback_and_restaging(self):
        """Producer-shaped transactions: compute_unbound -> stage_products [-> bind_plan -> set_ready] then
        commit (restore cleared), hold (cancel after FREEZE) or unsent (cancel after BIND); after a cancel the
        same frame is re-staged from the restored history and must account identically."""
        session = switch.install()
        rng = random.Random(31)
        producer = producer_state({("a", "b"): {"k": 1.0}})
        keys = [(f"o{i}", f"o{j}") for i in range(6) for j in range(i + 1, 6)]
        frame = 0
        outcomes = {"commit": 0, "hold": 0, "unsent": 0, "restaged": 0}
        while frame < 120:
            history_before = repr(sorted((k, v.iou) for k, v in producer.conflict_state.pairs.items()))
            restore = {"ages": dict(producer.tracker._ages), "last": dict(producer.tracker._last_time),
                       "conflict": snapshot_of(producer)}
            state_rng = random.Random(frame)
            updates = advance(producer, state_rng, frame, keys)
            evidence = {"rows": [producer.conflict_state.decisions[k] for k in updates]}
            staged = Mutable(evidence, {"frame": {"id": frame}, "_restore": restore})
            row, owner = new_row(staged), _owner()
            action = rng.choice(["commit", "commit", "hold", "unsent"])
            run_stage("compute_unbound", owner, row)
            row.products = (evidence, np.zeros(4))
            run_stage("stage_products", owner, row)
            if action in ("commit", "unsent"):
                row.plan = b'{"frame":%d}' % frame
                run_stage("bind_plan", owner, row)
                row.ready = READY
                run_stage("set_ready", owner, row)
            outcomes[action] += 1
            if action == "commit":
                restore.clear()                       # producer.commit: staged._restore = {}
                same(session, row.plan, staged, row.products)
                frame += 1
            else:
                NS.NumericSourceProducer._restore_history(producer, restore)
                self.assertEqual(repr(sorted((k, v.iou) for k, v in producer.conflict_state.pairs.items())),
                                 history_before)
                same(session, None, {"_restore": {"conflict": snapshot_of(producer)}}, None)
                if rng.random() < 0.5:                # re-stage the same frame after rollback
                    outcomes["restaged"] += 1
                    continue
                frame += 1
        self.assertTrue(all(outcomes.values()), outcomes)
        self.assertGreater(session.snapshot()["owner_shortcut_hits"], 0)

    def test_v10_accounting_timer_wrapper_keeps_exactness_and_shortcut(self):
        """A V10-derived prep entry wraps the session in the (read-only imported) V10 AccountingTimer."""
        from rl_agent.splitfusion_m3_route_b_v10.accounting_timing import AccountingTimer
        session = switch.install()
        timer = AccountingTimer(session, 64)
        OWN.payload_storage_bytes = timer
        try:
            producer = producer_state({("a", "b"): {"k": 1.0}})
            advance(producer, random.Random(4), 0, [("a", "b"), ("a", "c")])
            staged, products, _backing = random_staged(random.Random(4), snapshot_of(producer))
            row, owner = new_row(staged), _owner()
            for stage in ("compute_unbound", "stage_products", "bind_plan", "set_ready"):
                if stage == "stage_products":
                    row.products = products
                if stage in ("bind_plan", "set_ready"):
                    row.plan = b"plan"
                if stage == "set_ready":
                    row.ready = READY
                run_stage(stage, owner, row)
        finally:
            OWN.payload_storage_bytes = session
            switch.uninstall()
        self.assertEqual(session.snapshot()["owner_shortcut_hits"], 2)
        self.assertEqual(timer.calls, 4)
        self.assertEqual(timer.foreign_frames, 0)
        self.assertEqual(timer.caller_names, ["compute_unbound", "stage_products", "bind_plan", "set_ready"])

    def test_worker_entry_refuses_without_capability(self):
        from . import worker_entry as W
        saved = A.FAST_PATH_AVAILABLE
        with tempfile.TemporaryDirectory() as directory:
            try:
                A.FAST_PATH_AVAILABLE = False
                with self.assertRaises(RuntimeError) as raised:
                    W.main(["--role", "prep", "--trace-dir", directory])
                self.assertIn("REFUSED", str(raised.exception))
            finally:
                A.FAST_PATH_AVAILABLE = saved
            import json
            record = json.loads((Path(directory) / "storage_accounting_binding.json").read_text())
            self.assertTrue(record["restored"])
            self.assertIsNone(record["installed"])
            self.assertIs(OWN.payload_storage_bytes, OLD)


if __name__ == "__main__":
    unittest.main()

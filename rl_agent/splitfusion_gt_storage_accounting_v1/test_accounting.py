"""Targeted exactness tests: candidate vs original ``payload_storage_bytes`` (CPU only, no fixtures on disk).

  cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest -v \\
      rl_agent.splitfusion_gt_storage_accounting_v1.test_accounting
"""
from dataclasses import dataclass, field
import random
from types import SimpleNamespace

import numpy as np
import unittest

from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as CF
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS

from . import accounting as A, switch

OLD, NEW = OWN.payload_storage_bytes, A.payload_storage_bytes
PH = CF._PairHistory


def both(*values):
    old, new = OLD(*values), NEW(*values)
    assert old == new, (old, new)
    return old


def outcome(fn, *values):
    try:
        return ("ok", fn(*values))
    except BaseException as error:
        return ("raise", type(error), str(error))


@dataclass
class Mutable:
    a: object = None
    b: object = None


@dataclass(frozen=True)
class Frozen:
    a: object = None
    b: tuple = ()


# ------------------------------------------------------------------ aliasing / sharing
def test_shared_child_tuples_counted_once_in_any_order():
    t = (1.0, 2.0, 3.0)
    s = ("a", "bb")
    d = {"x": 1.0, "y": "z"}
    for values in ([t, t], [(t, t)], [{"k": t, "j": t}], [[t, s, d], (s, d, t)], [d, [d, d]], [Frozen(t, t), t]):
        both(*values)
        both(*reversed(values))
    assert both(t, t) == both(t) and both(d, d) == both(d)


def test_shared_empty_tuple_and_empty_containers():
    e = ()
    both(e, e, (), [(), ()], {"a": ()}, PH(), PH(), [], set(), frozenset(), {})
    both([PH(), PH()], (e,))
    assert both(()) == 16 and both([]) == 16 and both({}) == 16


def test_pair_history_versions_share_best_tuple():
    p0 = PH()
    p1 = p0.appended(0.8, 0.1, 0.2, "s1")
    p2 = p1.appended(0.1, 0.3, 0.4, "s2")        # max keeps p1.best: the SAME tuple object is reused
    assert p2.best is p1.best
    p3 = p2.appended(0.9, 0.2, 0.1, "s3")         # new best tuple
    for values in ([p1, p2], [p2, p1], [{"k": p1}, {"k": p2}], [p1, p2, p3], [(p3, p2, p1)], [p2.best, p1, p2]):
        both(*values)
        both(*reversed(values))


def test_multiple_top_level_arguments_and_repeats():
    t = (1.0,) * 40
    staged = {"a": t, "b": [t, {"c": t}], "s": {"x", "y"}}
    for args in ((b"plan", staged, None), (None, staged, staged), (staged, b"plan"), (staged["a"], staged)):
        both(*args)


def _random_structure(rng, pool, depth=0):
    choice = rng.random()
    if pool and choice < 0.15:
        return rng.choice(pool)                                       # alias an existing container
    if depth > 4 or choice < 0.45:
        return rng.choice([None, True, 0, -3, 2.5, float("nan"), np.float64(1.5), np.float32(2), np.int64(7),
                           np.uint8(3), np.bool_(True), np.int32(-1), "", "ascii", "ünï", "日本語", b"raw",
                           bytearray(b"ab"), memoryview(b"xyz"), np.zeros(3), np.arange(4, dtype=np.int16)])
    n = rng.randrange(0, 6)
    kind = rng.randrange(7)
    if kind == 0:
        value = [_random_structure(rng, pool, depth + 1) for _ in range(n)]
    elif kind == 1:
        value = tuple(_random_structure(rng, pool, depth + 1) for _ in range(n))
    elif kind == 2:
        value = {f"k{i}": _random_structure(rng, pool, depth + 1) for i in range(n)}
    elif kind == 3:
        value = {rng.choice(["a", "b", "c", 1, 2.0, None, ("t", 1)]) for _ in range(n)}
    elif kind == 4:
        value = Mutable(_random_structure(rng, pool, depth + 1), _random_structure(rng, pool, depth + 1))
    elif kind == 5:
        value = PH(*(tuple(sorted(rng.random() for _ in range(n))) for _ in range(3)), n,
                   (rng.random(), f"s{n}"))
    else:
        value = [rng.random() for _ in range(n)] if rng.random() < 0.5 else {f"f{i}": rng.random() for i in range(n)}
    pool.append(value)
    return value


def test_randomized_differential_with_aliasing():
    rng = random.Random(20261008)
    for _ in range(3000):
        pool = []
        values = [_random_structure(rng, pool) for _ in range(rng.randrange(1, 4))]
        both(*values)
        both(*reversed(values))


# ------------------------------------------------------------------ exact-type boundaries and rejection rules
class ListSub(list):
    def __len__(self):
        return 999


class StrSub(str):
    def encode(self, *args, **kwargs):
        return b"x" * 50


class FloatSub(float):
    pass


class DictSub(dict):
    pass


def test_subclasses_take_original_path():
    both(ListSub([1.0, 2.0]), [StrSub("ab"), "cd"], (FloatSub(1.0), 2.0), DictSub(a=1.0), {"a": StrSub("q")},
         {StrSub("k"): 1}, [np.str_("np"), "s"], (np.bytes_(b"b"),), [np.complex64(1)], [np.longdouble(1)])


def test_rejections_identical():
    cases = [[1.0, np.array([object()], dtype=object)], {"a": 1.0, "b": object()}, (1.0, object()),
             ["ok", "\ud800"], {"\udfff": 1.0}, {"\udfff", "a"}, ("a", "b\ud800c"), [1, 2, lambda: 0], [Ellipsis]]
    for value in cases:
        assert outcome(OLD, value) == outcome(NEW, value)
        assert outcome(OLD, b"x", value) == outcome(NEW, b"x", value)


def test_function_closures_and_ndarray_views():
    a = np.arange(12.0).reshape(3, 4)
    masks = {"car": a[:1], "person": a}
    def same(tag, _m=masks):                       # closure-owned dict, as Staged.same_class
        return _m[tag]
    def closure():
        return masks, a
    both(closure, masks, a, a[1:])
    both(Mutable(closure, (a, a.T)))


# ------------------------------------------------------------------ rollback / commit / cancel on real conflict code
def test_conflict_snapshot_rollback_sequence():
    """Real _PairHistory.appended + the literal producer _conflict_snapshot/_restore_history bodies."""
    rng = random.Random(7)
    state = SimpleNamespace(templates={("a", "b"): {"k": 1.0}}, pairs={}, decisions={}, frame_count=0, row_count=0,
                            previous_frame=-1, previous_timestamp=-1.0, first_timestamp=None, episode=None,
                            seen_samples=set(), chain="c0")
    producer = SimpleNamespace(conflict_state=state, tracker=SimpleNamespace(_ages={}, _last_time={}))
    keys = [(f"o{i}", f"o{j}") for i in range(6) for j in range(i + 1, 6)]
    for frame in range(60):
        restore = {"ages": dict(producer.tracker._ages), "last": dict(producer.tracker._last_time),
                   "conflict": NS.NumericSourceProducer._conflict_snapshot(producer)}
        before = both({"_restore": restore})
        updates = {}
        for key in rng.sample(keys, rng.randrange(1, 6)):
            updates[key] = state.pairs.get(key, PH()).appended(rng.random(), rng.random(), rng.random(), f"s{frame}")
        state.pairs.update(updates)                # derive: new history versions
        state.templates.update({k: {"geom": rng.random(), "name": "n"} for k in updates if k not in state.templates})
        state.seen_samples.add(f"s{frame}")
        staged = {"conflict": {"q": [list(k) for k in updates]}, "_restore": restore}
        both(b"plan", staged, None)                # FREEZE/BIND _account while uncommitted
        both(b"plan", staged, {"products": tuple(updates.values())})
        if frame % 3 == 0:                         # cancel: exact rollback, accounting returns to its pre-frame value
            NS.NumericSourceProducer._restore_history(producer, restore)
            assert both({"_restore": {"ages": {}, "last": {},
                                      "conflict": NS.NumericSourceProducer._conflict_snapshot(producer)}}) == before
        else:                                      # commit: rollback state is released
            staged["_restore"] = {}
            both(b"plan", staged, None)


# ------------------------------------------------------------------ capacity boundary through the real owner method
def _row(staged, reservation):
    return SimpleNamespace(plan=b"plan-bytes", staged=staged, products=None, ready=None, reservation=reservation,
                           payload_bytes=None)


def _owner():
    owner = SimpleNamespace(_failure=None)
    owner._fail = lambda message, exc_type=OWN.OwnershipError: OWN.ProducerOwner._fail(owner, message, exc_type)
    return owner


def capacity_boundary(delta):
    staged = {"pairs": {("a", "b"): PH().appended(0.5, 0.1, 0.1, "s")}, "rows": [{"x": 1.0}] * 3, "img": np.zeros(10)}
    exact = OLD(b"plan-bytes", staged, None)
    results = []
    for installed in (False, True):
        if installed:
            switch.install()
        try:
            row = _row(staged, exact + delta)
            results.append(outcome(OWN.ProducerOwner._account, _owner(), row) + (row.payload_bytes,))
        finally:
            switch.uninstall()
    assert results[0] == results[1]
    assert (results[0][0] == "ok") == (delta == 0)
    if delta:
        assert results[0][1] is OWN.CapacityError


# ------------------------------------------------------------------ switch
def test_switch_reversible_and_guarded():
    assert OWN.payload_storage_bytes is switch.ORIGINAL_PAYLOAD_STORAGE_BYTES and not switch.installed()
    switch.install()
    try:
        assert switch.installed() and OWN.payload_storage_bytes is A.payload_storage_bytes
        try:
            switch.install()
            raise AssertionError("second install accepted")
        except RuntimeError:
            pass
    finally:
        switch.uninstall()
    assert OWN.payload_storage_bytes is switch.ORIGINAL_PAYLOAD_STORAGE_BYTES and not switch.installed()
    OWN.payload_storage_bytes = lambda *v: 0
    try:
        try:
            switch.install()
            raise AssertionError("foreign accounting function accepted")
        except RuntimeError:
            pass
    finally:
        switch.uninstall()
        OWN.payload_storage_bytes = switch.ORIGINAL_PAYLOAD_STORAGE_BYTES


def test_no_state_retained_between_calls():
    names = {n for n in vars(A) if not n.startswith("__")}
    t = tuple(float(i) for i in range(100))
    NEW(t, {"a": t})
    assert {n for n in vars(A) if not n.startswith("__")} == names
    assert not any(isinstance(v, (dict, list, set)) and v for n, v in vars(A).items()
                   if not n.startswith("_") and n not in ("VERSION",))


class AccountingTests(unittest.TestCase):
    def test_shared_child_tuples(self): test_shared_child_tuples_counted_once_in_any_order()
    def test_shared_empty(self): test_shared_empty_tuple_and_empty_containers()
    def test_pair_history_best(self): test_pair_history_versions_share_best_tuple()
    def test_top_level(self): test_multiple_top_level_arguments_and_repeats()
    def test_randomized(self): test_randomized_differential_with_aliasing()
    def test_subclasses(self): test_subclasses_take_original_path()
    def test_rejections(self): test_rejections_identical()
    def test_closures(self): test_function_closures_and_ndarray_views()
    def test_rollback(self): test_conflict_snapshot_rollback_sequence()
    def test_capacity_exact_fits(self): capacity_boundary(0)
    def test_capacity_one_byte_over(self): capacity_boundary(-1)
    def test_switch(self): test_switch_reversible_and_guarded()
    def test_stateless(self): test_no_state_retained_between_calls()


if __name__ == "__main__":
    unittest.main()

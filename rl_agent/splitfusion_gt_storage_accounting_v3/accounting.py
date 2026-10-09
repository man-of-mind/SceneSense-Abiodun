"""Exact per-object reachable-set accounting for ``ownership.payload_storage_bytes`` (offline candidate V3).

The logical byte definition is the original one (``splitfusion_numeric_gt_integration_v2.ownership``). Outside the
producer rollback snapshot the traversal is the unchanged cache-free V1 traversal. Inside it, V2's per-snapshot
signatures, pins and context-keyed subtotals are replaced by a per-call computation:

Value identity (why the integer is the original one)
  Under the original semantics each distinct reachable container is counted once -- its own size is its header,
  8 per element and the sizes of its direct scalar children -- and each scalar is counted per reference. A snapshot
  ``S`` of the validated shape (below) reaches only containers whose children are scalars or other members of the
  same shape, so none of them can contain ``S`` or any ancestor of ``S``. When the original reaches ``S`` with a
  visited set ``seen``, every member of the reachable set ``R`` that is already in ``seen`` was traversed
  completely earlier in the same call, together with everything it reaches. The original therefore adds exactly
  ``own(S) + sum(own(c) for c in R - seen)`` and leaves ``seen | R``. V3 computes that set difference afresh on
  every call: there is no cached snapshot subtotal and no cached alias context.

Validated snapshot shape (anything else -> unchanged exact traversal of that dict, from the same ``seen``)
  exact ``dict`` with exactly the 11 producer keys, all exact ``str``; ``templates``/``pairs``/``decisions`` exact
  ``dict`` with exact ``tuple`` keys whose elements are exact scalars; template/decision rows exact ``dict`` whose
  keys and values are exact scalars; pair values exact ``_PairHistory`` (unchanged class, guarded per call) whose
  ``iou``/``ranges``/``near``/``best`` are exact ``tuple`` of exact scalars and whose ``overlap_count`` is an exact
  numeric scalar; ``seen_samples`` exact ``set`` of exact scalars; the other 7 values exact scalars. "Exact scalar"
  is V1's ``_SCALAR`` whitelist of static immutable types (numeric, ``str``, ``None``).

Per-object own sizes (no value equality anywhere)
  * row dicts: memoised under their CPython dict version tag. In CPython 3.10 every dict creation and every
    mutation takes a fresh value of one global counter, so a tag identifies one dict in one state; the state fixes
    its key/value objects, which were validated as immutable scalars when the entry was made. Tags are re-read
    on every call. The tag semantics are re-proved by ``capabilities()`` before the fast path is enabled.
  * tuples (pair keys, PairHistory fields): memoised by identity while strongly pinned (the pin prevents id reuse);
    a tuple is immutable and its elements were validated as immutable scalars.
  * ``_PairHistory``: own size 16 + 16, derived each call from the exact field types; its four tuple fields are
    read afresh each call (C-level ``attrgetter``), so a field rebound by ``object.__setattr__`` is a new child.
  * the three maps, the sample set and ``S`` itself are recomputed each call.
  The memo holds only objects reachable from the most recent snapshot that needed a new entry (sweep on every
  rebuild); its size is bounded by ``max_objects``. A snapshot whose reachable count could exceed the bound takes
  the exact traversal (fail closed); nothing is evicted silently.

Owner-stage shortcut (``bind_plan``/``set_ready``)
  Reuses the previous owner result only when the arguments are the same ``staged`` and ``products`` objects, the
  plan is an exact immutable ``None``/``bytes``/``str`` (integer delta), and a whole-graph witness recorded by the
  base call is unchanged: every reached dict's version tag (incl. ``S``, its maps, rows and mappingproxy
  backing dicts), every list/set's length and element identities (elements pinned), every ndarray's
  ``(dtype.hasobject, nbytes)`` -- exactly what the original reads from an ndarray; ``ndarray.__setstate__`` can
  change the dtype in place at equal identity and nbytes (review_01 counterexample, fixed in .v2) --
  every dataclass instance's type and field-value identities, every closure's cell-content identities, and every
  ``_PairHistory``'s instance-dict tag under the class guard. Any reached object outside those exact kinds
  (subclasses, other mappings, ``bytearray``/``memoryview``, non-whitelisted scalar types) makes the witness
  ineligible and the call is a full traversal.

Capability
  Without the CPython dict mutation tag (or if its semantics, the mappingproxy referent layout or the
  ``_PairHistory`` layout cannot be re-proved) the session runs the exact V1 traversal only; the prep-worker entry
  refuses to start in that case. Rejection rules (object ndarray, unsupported type, lone surrogates) are those of
  the unchanged traversal, which also runs whenever the fast path declines.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import gc
from itertools import chain, compress, repeat
from operator import attrgetter, is_
import sys
from types import FunctionType, MappingProxyType

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1.conflicts import _PairHistory
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_integration_v2.ownership import OwnershipError
from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as V1

try:
    from _testcapi import dict_get_version as _dict_version
except ImportError:  # Fail closed on interpreters without CPython's mutation tag.
    _dict_version = None

VERSION = "splitfusion_gt_storage_accounting_v3.per_object_reachable_set.v2"
DEFAULT_MAX_OBJECTS = 32768
_SCALAR = V1._SCALAR
_NUMERIC = V1._NUMERIC
_CONTAINERS = (list, tuple, set, frozenset)
_flat = V1._flat
_DICT_ONLY = frozenset((dict,))
_TUPLE_ONLY = frozenset((tuple,))
_PH_ONLY = frozenset((_PairHistory,))
_STR_ONLY = frozenset((str,))
_PLAN_TYPES = (type(None), bytes, str)
_SNAPSHOT_KEYS = frozenset(("templates", "pairs", "decisions", "frame_count", "row_count", "previous_frame",
                            "previous_timestamp", "first_timestamp", "episode", "seen_samples", "chain"))
_SNAPSHOT_SCALARS = ("frame_count", "row_count", "previous_frame", "previous_timestamp",
                     "first_timestamp", "episode", "chain")
_SNAPSHOT_OWN_FRAMING = 16 + 8 * len(_SNAPSHOT_KEYS)
_PH_FIELD_NAMES = ("iou", "ranges", "near", "overlap_count", "best")
_PH_TUPLES = attrgetter("iou", "ranges", "near", "best")
_PH_COUNT = attrgetter("overlap_count")
_PH_OWN = 16 + 16          # dataclass header + numeric overlap_count; the four tuple fields are separate containers
def _array_state(array):
    """What the original reads from an ndarray: the object-dtype refusal and the visible byte count."""
    return array.dtype.hasobject, array.nbytes
_OWNER_CHAINS = frozenset((("stage_products", "bind_plan"), ("bind_plan", "set_ready"),
                           ("stage_products", "set_ready")))
_WITNESS_CALLERS = frozenset(("stage_products", "bind_plan"))


# --------------------------------------------------------------------------- capability (re-proved, never assumed)
def _probe_dict_tags():
    """Re-prove the CPython 3.10 dict-version properties relied on: one strictly increasing global counter taken by
    every creation and every mutation (incl. instance dicts written by ``object.__setattr__``/``__delattr__``)."""
    if _dict_version is None:
        return "CPython dict_get_version unavailable"
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 10):
        return f"dict tag semantics are proved for CPython 3.10 only, not {sys.implementation.name} {sys.version}"
    tags = []

    def step(mapping):
        tag = _dict_version(mapping)
        if tags and tag <= tags[-1]:
            raise AssertionError(f"dict tag did not advance globally ({tags[-1]} -> {tag})")
        tags.append(tag)

    class _Probe:
        pass

    try:
        a, first, second = {}, object(), object()
        step(a)
        b = {}
        step(b)
        a["k"] = first; step(a)
        a["k"] = second; step(a)                    # same key, different value object
        a.update(j=1); step(a)
        a.setdefault("m", 2); step(a)
        a.pop("m"); step(a)
        del a["j"]; step(a)
        a.popitem(); step(a)
        a["x"] = 1; a.clear(); step(a)
        step(a.copy())
        step(dict(b))
        b |= {"z": 3}; step(b)
        step(dict.fromkeys("ab"))
        step({k: 1 for k in "cd"})
        instance = _Probe()
        instance.v = 1; step(vars(instance))
        object.__setattr__(instance, "v", first); step(vars(instance))
        object.__delattr__(instance, "v"); step(vars(instance))
        unchanged = {"q": first}
        tag = _dict_version(unchanged)
        unchanged["q"] = first                       # identical object: no state change, tag may stay
        _ = unchanged.get("q"), list(unchanged.items())
        if _dict_version(unchanged) < tag:
            raise AssertionError("dict tag decreased")
    except Exception as error:
        return f"dict tag semantics not proved: {type(error).__name__}: {error}"
    return None


def _probe_mappingproxy():
    backing = {"a": 1}
    referents = gc.get_referents(MappingProxyType(backing))
    return None if len(referents) == 1 and referents[0] is backing else "mappingproxy backing mapping not resolvable"


def _class_dict(cls):
    referents = gc.get_referents(vars(cls))
    return referents[0] if len(referents) == 1 and type(referents[0]) is dict else None


def _probe_pair_history():
    cls = _PairHistory
    if tuple(field.name for field in fields(cls)) != _PH_FIELD_NAMES:
        return "unexpected _PairHistory field layout"
    if cls.__mro__ != (cls, object) or "__slots__" in vars(cls):
        return "unexpected _PairHistory class layout"
    for name in ("__getattribute__", "__getattr__"):
        if name in vars(cls):
            return "_PairHistory overrides attribute access"
    for name in _PH_FIELD_NAMES:
        attribute = vars(cls).get(name)
        if attribute is not None and (hasattr(type(attribute), "__set__") or hasattr(type(attribute), "__delete__")):
            return "_PairHistory field is a data descriptor"
    probe = cls().appended(.5, .25, .125, "probe")
    if type(getattr(probe, "__dict__", None)) is not dict or set(vars(probe)) != set(_PH_FIELD_NAMES):
        return "_PairHistory instances do not keep fields in an instance dict"
    if _class_dict(cls) is None or type(cls.__dataclass_fields__) is not dict:
        return "_PairHistory class dictionaries not resolvable"
    return None


def capabilities():
    """Every precondition of the fast path; all values must be None (no reason) for it to be enabled."""
    reasons = dict(dict_tags=_probe_dict_tags(), mappingproxy=None, pair_history=None)
    if reasons["dict_tags"] is None:
        reasons["mappingproxy"] = _probe_mappingproxy()
        reasons["pair_history"] = _probe_pair_history()
    return reasons


_CAPABILITY = capabilities()
FAST_PATH_AVAILABLE = all(reason is None for reason in _CAPABILITY.values())
if FAST_PATH_AVAILABLE:
    _PH_CLASS_DICT = _class_dict(_PairHistory)
    _PH_CLASS_TAG = _dict_version(_PH_CLASS_DICT)
    _PH_FIELDS_DICT = _PairHistory.__dataclass_fields__
    _PH_FIELDS_TAG = _dict_version(_PH_FIELDS_DICT)
    _PH_MRO = _PairHistory.__mro__
else:
    _PH_CLASS_DICT = _PH_FIELDS_DICT = _PH_MRO = None
    _PH_CLASS_TAG = _PH_FIELDS_TAG = None


def _pair_history_class_unchanged():
    """O(1) per-call guard: class dict, dataclass field table and MRO exactly as proved at import."""
    return (_dict_version(_PH_CLASS_DICT) == _PH_CLASS_TAG and _dict_version(_PH_FIELDS_DICT) == _PH_FIELDS_TAG
            and _PairHistory.__mro__ is _PH_MRO and _class_dict(_PairHistory) is _PH_CLASS_DICT)


def _plan_size(value):
    """Original size of an exact immutable plan scalar, or None (no shortcut)."""
    kind = type(value)
    if kind is bytes:
        return 8 + len(value)
    if kind is str:
        try:
            return 8 + len(value.encode("utf-8"))
        except Exception:
            return None
    return 1 if value is None else None


def _dataclass_children(value):
    return tuple(getattr(value, field.name) for field in fields(value))


def _cell_contents(function):
    return tuple(cell.cell_contents for cell in (function.__closure__ or ()))


class _Witness:
    """Whole-graph mutation witness of one full owner call; see the module docstring."""

    __slots__ = ("dicts", "seqs", "proxy_dicts", "pair_histories", "arrays", "objects", "functions", "ineligible",
                 "dict_tags", "seq_lens", "seq_items", "proxy_tags", "pair_history_tags", "array_states")

    def __init__(self):
        self.dicts, self.seqs, self.proxy_dicts, self.pair_histories = [], [], [], []
        self.arrays, self.objects, self.functions = [], [], []
        self.ineligible = False
        self.dict_tags = self.seq_lens = self.seq_items = None
        self.proxy_tags = self.pair_history_tags = self.array_states = None

    def seal(self):
        if self.ineligible:
            return
        self.dict_tags = list(map(_dict_version, self.dicts))
        self.seq_lens = list(map(len, self.seqs))
        self.seq_items = tuple(chain.from_iterable(self.seqs))
        self.proxy_tags = list(map(_dict_version, self.proxy_dicts))
        self.pair_history_tags = list(map(_dict_version, map(vars, self.pair_histories)))
        self.array_states = list(map(_array_state, self.arrays))

    def unchanged(self):
        if self.ineligible or self.dict_tags is None:
            return False
        try:
            return (list(map(_dict_version, self.dicts)) == self.dict_tags
                    and list(map(len, self.seqs)) == self.seq_lens
                    and all(map(is_, chain.from_iterable(self.seqs), self.seq_items))
                    and list(map(_dict_version, self.proxy_dicts)) == self.proxy_tags
                    and (not self.pair_histories
                         or (_pair_history_class_unchanged()
                             and set(map(type, self.pair_histories)) <= _PH_ONLY
                             and list(map(_dict_version, map(vars, self.pair_histories))) == self.pair_history_tags))
                    and list(map(_array_state, self.arrays)) == self.array_states
                    and all(type(value) is kind and all(map(is_, _dataclass_children(value), children))
                            and len(children) == len(fields(value))
                            for value, kind, children in self.objects)
                    and all(len(contents) == len(function.__closure__ or ())
                            and all(map(is_, _cell_contents(function), contents))
                            for function, contents in self.functions))
        except Exception:
            return False

    def size(self):
        return dict(dicts=len(self.dicts), sequences=len(self.seqs), sequence_items=len(self.seq_items or ()),
                    mappingproxies=len(self.proxy_dicts), pair_histories=len(self.pair_histories),
                    arrays=len(self.arrays), dataclasses=len(self.objects), functions=len(self.functions),
                    ineligible=self.ineligible)


class _CallMemo:
    """Entries for every memoised object reachable from the snapshots of one call (replaces the memo)."""

    __slots__ = ("tuple_owns", "tuple_pins", "row_owns", "misses")

    def __init__(self):
        self.tuple_owns, self.tuple_pins, self.row_owns, self.misses = {}, {}, {}, 0


class AccountingSession:
    """Process-local callable installed only in the GT preparation worker."""

    def __init__(self, max_objects=DEFAULT_MAX_OBJECTS):
        if type(max_objects) is not int or max_objects < 1:
            raise ValueError("positive bounded accounting memo capacity required")
        self.max_objects = max_objects
        self.fast_path = FAST_PATH_AVAILABLE
        self._tuple_owns, self._tuple_pins, self._row_owns = {}, {}, {}
        self._last_owner_call = None
        self._closed = False
        self.calls = self.failed_calls = self.full_calls = 0
        self.snapshot_fast = self.snapshot_fallbacks = self.snapshot_bound_fallbacks = 0
        self.tuple_hits = self.tuple_misses = self.row_hits = self.row_misses = 0
        self.memo_rebuilds = self.memo_bound_discards = 0
        self.owner_shortcut_hits = self.owner_shortcut_rejections = 0
        self.witnesses = self.witness_ineligible = 0
        self.peak_memo_objects = 0
        self.fallback_reasons = {}

    # ------------------------------------------------------------------ owner stage identification (as V2)
    @staticmethod
    def _owner_caller():
        """Return the retained owner stage, or None for every other caller."""
        frame = sys._getframe(1)
        code = OWN.ProducerOwner._account.__code__
        for _ in range(8):
            if frame is None:
                break
            if frame.f_code is code:
                return frame.f_back.f_code.co_name if frame.f_back is not None else None
            frame = frame.f_back
        return None

    def _decline(self, reason):
        self.snapshot_fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        return None

    # ------------------------------------------------------------------ snapshot part
    def _snapshot(self, value, seen, witness, memo):
        """own(S) + sum(own(c) for c in R - seen), seen |= R; or None (no side effect) -> exact traversal."""
        if not _pair_history_class_unchanged():
            return self._decline("pair_history_class_changed")
        templates, pairs, decisions, samples = (value["templates"], value["pairs"], value["decisions"],
                                                value["seen_samples"])
        if type(templates) is not dict or type(pairs) is not dict or type(decisions) is not dict \
                or type(samples) is not set:
            return self._decline("snapshot_member_type")
        if len(templates) + len(decisions) + len(templates) + len(pairs) + len(decisions) + 5 * len(pairs) + 4 \
                > self.max_objects:
            self.snapshot_bound_fallbacks += 1
            return self._decline("memo_bound")
        scalars = [value[name] for name in _SNAPSHOT_SCALARS]
        scalar_kinds = set(map(type, scalars))
        if not scalar_kinds <= _SCALAR:
            return self._decline("snapshot_scalar_type")
        rows = list(templates.values())
        rows += decisions.values()
        if not set(map(type, rows)) <= _DICT_ONLY:
            return self._decline("row_type")
        histories = list(pairs.values())
        if not set(map(type, histories)) <= _PH_ONLY:
            return self._decline("pair_history_type")
        if not set(map(type, map(_PH_COUNT, histories))) <= _NUMERIC:
            return self._decline("overlap_count_type")
        tuples = list(templates)
        tuples += pairs
        tuples += decisions
        tuples += chain.from_iterable(map(_PH_TUPLES, histories))
        if not set(map(type, tuples)) <= _TUPLE_ONLY:
            return self._decline("key_or_field_type")
        sample_list = list(samples)
        sample_kinds = set(map(type, sample_list))
        if not sample_kinds <= _SCALAR:
            return self._decline("sample_type")
        try:
            own_samples = 16 + 8 * len(sample_list) + _flat(sample_list, sample_kinds)
            own_snapshot = _SNAPSHOT_OWN_FRAMING + _flat(list(value), _STR_ONLY) + _flat(scalars, scalar_kinds)
        except Exception:
            return self._decline("scalar_encoding")

        tuple_ids = list(map(id, tuples))
        tuple_owns = list(map(self._tuple_owns.get, tuple_ids))
        tuple_misses = 0
        if None in tuple_owns:
            for index in list(compress(range(len(tuple_owns)), map(is_, tuple_owns, repeat(None)))):
                item = tuples[index]
                elements = list(item)
                kinds = set(map(type, elements))
                if not kinds <= _SCALAR:
                    return self._decline("tuple_element_type")
                try:
                    tuple_owns[index] = 16 + 8 * len(elements) + _flat(elements, kinds)
                except Exception:
                    return self._decline("scalar_encoding")
                tuple_misses += 1
        row_tags = list(map(_dict_version, rows))
        row_owns = list(map(self._row_owns.get, row_tags))
        row_misses = 0
        if None in row_owns:
            for index in list(compress(range(len(row_owns)), map(is_, row_owns, repeat(None)))):
                row = rows[index]
                keys, items = list(row), list(row.values())
                key_kinds, item_kinds = set(map(type, keys)), set(map(type, items))
                if not (key_kinds <= _SCALAR and item_kinds <= _SCALAR):
                    return self._decline("row_value_type")
                try:
                    row_owns[index] = 16 + 8 * len(keys) + _flat(keys, key_kinds) + _flat(items, item_kinds)
                except Exception:
                    return self._decline("scalar_encoding")
                row_misses += 1

        own_by_id = dict(zip(tuple_ids, tuple_owns))
        own_by_id.update(zip(map(id, rows), row_owns))
        own_by_id.update(zip(map(id, histories), repeat(_PH_OWN)))
        own_by_id[id(templates)] = 16 + 8 * len(templates)
        own_by_id[id(pairs)] = 16 + 8 * len(pairs)
        own_by_id[id(decisions)] = 16 + 8 * len(decisions)
        own_by_id[id(samples)] = own_samples
        total = own_snapshot + sum(map(own_by_id.__getitem__, own_by_id.keys() - seen))

        # Infallible bookkeeping only from here on.
        memo.tuple_owns.update(zip(tuple_ids, tuple_owns))
        memo.tuple_pins.update(zip(tuple_ids, tuples))
        memo.row_owns.update(zip(row_tags, row_owns))
        memo.misses += tuple_misses + row_misses
        self.tuple_misses += tuple_misses
        self.tuple_hits += len(tuple_ids) - tuple_misses
        self.row_misses += row_misses
        self.row_hits += len(row_tags) - row_misses
        if witness is not None:
            witness.dicts.append(templates)
            witness.dicts.append(pairs)
            witness.dicts.append(decisions)
            witness.dicts.extend(rows)
            witness.seqs.append(samples)
            witness.pair_histories.extend(histories)
        seen.update(own_by_id)
        self.snapshot_fast += 1
        return total

    def _install(self, memo):
        if not memo.misses:
            return                          # every object was already memoised; the memo is unchanged
        objects = len(memo.tuple_owns) + len(memo.row_owns)
        if objects > self.max_objects:
            self.memo_bound_discards += 1
            self._tuple_owns, self._tuple_pins, self._row_owns = {}, {}, {}
            return
        self._tuple_owns, self._tuple_pins, self._row_owns = memo.tuple_owns, memo.tuple_pins, memo.row_owns
        self.memo_rebuilds += 1
        self.peak_memo_objects = max(self.peak_memo_objects, objects)

    # ------------------------------------------------------------------ complete call
    def _traverse(self, values, witness, memo):
        seen = set()
        fast = self.fast_path
        record = witness is not None
        snapshot = self._snapshot

        def size(value):
            if value is None:
                return 1
            if isinstance(value, (str, bytes, bytearray, memoryview)):
                if record and type(value) is not str and type(value) is not bytes:
                    witness.ineligible = True
                return 8 + (len(value.encode("utf-8")) if isinstance(value, str) else
                            value.nbytes if isinstance(value, memoryview) else len(value))
            if isinstance(value, (bool, int, float, np.generic)):
                if record and type(value) not in _NUMERIC:
                    witness.ineligible = True
                return 16
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise OwnershipError("object ndarray cannot be retained")
                if record:
                    if type(value) is np.ndarray:
                        witness.arrays.append(value)
                    else:
                        witness.ineligible = True
                return 32 + value.nbytes
            if isinstance(value, Mapping):
                if type(value) is dict:
                    if record:
                        witness.dicts.append(value)
                    if fast and len(value) == 11 and set(map(type, value)) <= _STR_ONLY \
                            and value.keys() == _SNAPSHOT_KEYS:
                        part = snapshot(value, seen, witness, memo)
                        if part is not None:
                            return part
                    keys, items = list(value), list(value.values())
                    key_kinds, item_kinds = set(map(type, keys)), set(map(type, items))
                    if key_kinds <= _SCALAR and item_kinds <= _SCALAR:
                        try:
                            return 16 + 8 * len(keys) + _flat(keys, key_kinds) + _flat(items, item_kinds)
                        except Exception:     # the original loop raises it exactly
                            pass
                elif record:
                    backing = gc.get_referents(value) if type(value) is MappingProxyType else ()
                    if len(backing) == 1 and type(backing[0]) is dict:
                        witness.proxy_dicts.append(backing[0])
                    else:
                        witness.ineligible = True
                return 16 + sum(8 + size(key) + size(item) for key, item in value.items())
            if isinstance(value, (list, tuple, set, frozenset)):
                kind = type(value)
                if record:
                    if kind is list or kind is set:
                        witness.seqs.append(value)
                    elif kind is not tuple and kind is not frozenset:
                        witness.ineligible = True
                if kind in _CONTAINERS:
                    elements = list(value)
                    kinds = set(map(type, elements))
                    if kinds <= _SCALAR:
                        try:
                            return 16 + 8 * len(elements) + _flat(elements, kinds)
                        except Exception:     # the original loop raises it exactly
                            pass
                return 16 + sum(8 + size(item) for item in value)
            if is_dataclass(value):
                if not record:
                    return 16 + sum(size(getattr(value, field.name)) for field in fields(value))
                children, total = [], 16
                for field in fields(value):  # original order: read one field, then traverse it
                    child = getattr(value, field.name)
                    children.append(child)
                    total += size(child)
                witness.objects.append((value, type(value), tuple(children)))
                return total
            if isinstance(value, FunctionType):
                if not record:
                    return 16 + sum(size(cell.cell_contents) for cell in (value.__closure__ or ()))
                contents, total = [], 16
                for cell in (value.__closure__ or ()):
                    child = cell.cell_contents
                    contents.append(child)
                    total += size(child)
                witness.functions.append((value, tuple(contents)))
                return total
            raise OwnershipError("unsupported retained payload type: " + type(value).__name__)

        return sum(size(value) for value in values)

    def __call__(self, *values):
        if self._closed:
            raise RuntimeError("accounting session is closed")
        self.calls += 1
        caller = self._owner_caller() if self.fast_path else None
        previous, self._last_owner_call = self._last_owner_call, None
        if (caller in ("bind_plan", "set_ready") and previous is not None
                and (previous[0], caller) in _OWNER_CHAINS and previous[3] is not None
                and len(values) == len(previous[1]) == 3
                and values[1] is previous[1][1] and values[2] is previous[1][2]):
            prior_size, current_size = _plan_size(previous[1][0]), _plan_size(values[0])
            if prior_size is not None and current_size is not None:
                if previous[3].unchanged():
                    result = previous[2] - prior_size + current_size
                    self.owner_shortcut_hits += 1
                    self._last_owner_call = (caller, values, result, previous[3])
                    return result
                self.owner_shortcut_rejections += 1
        witness = _Witness() if caller in _WITNESS_CALLERS and len(values) == 3 else None
        memo = _CallMemo()
        self.full_calls += 1
        try:
            result = self._traverse(values, witness, memo)
        except BaseException:
            self.failed_calls += 1           # memo untouched: every entry is exact independently of this call
            raise
        self._install(memo)
        if witness is not None:
            witness.seal()
            self.witnesses += 1
            self.witness_ineligible += int(witness.ineligible)
            if witness.ineligible:
                witness = None
        self._last_owner_call = (caller, values, result, witness) if caller is not None else None
        return result

    # ------------------------------------------------------------------ reporting and lifetime
    def memo_objects(self):
        return len(self._tuple_owns) + len(self._row_owns)

    def snapshot(self):
        witness = self._last_owner_call[3] if self._last_owner_call is not None else None
        retained = (sum(map(sys.getsizeof, (self._tuple_owns, self._tuple_pins, self._row_owns)))
                    + sum(map(sys.getsizeof, self._row_owns)) + sum(map(sys.getsizeof, self._tuple_owns))
                    + sum(map(sys.getsizeof, self._tuple_owns.values()))
                    + sum(map(sys.getsizeof, self._row_owns.values())))
        if witness is not None:
            retained += (sum(map(sys.getsizeof, (witness.dicts, witness.seqs, witness.proxy_dicts,
                                                 witness.pair_histories, witness.arrays, witness.objects,
                                                 witness.functions, witness.seq_items, witness.dict_tags,
                                                 witness.pair_history_tags)))
                         + sum(map(sys.getsizeof, witness.dict_tags)) + sum(map(sys.getsizeof, witness.pair_history_tags)))
        return dict(version=VERSION, fast_path=self.fast_path, capability=dict(_CAPABILITY),
                    max_objects=self.max_objects, memo_objects=self.memo_objects(),
                    memo_tuple_entries=len(self._tuple_owns), memo_pinned_tuples=len(self._tuple_pins),
                    memo_row_entries=len(self._row_owns), peak_memo_objects=self.peak_memo_objects,
                    calls=self.calls, full_calls=self.full_calls, failed_calls=self.failed_calls,
                    snapshot_fast=self.snapshot_fast, snapshot_fallbacks=self.snapshot_fallbacks,
                    snapshot_bound_fallbacks=self.snapshot_bound_fallbacks,
                    fallback_reasons=dict(self.fallback_reasons),
                    tuple_hits=self.tuple_hits, tuple_misses=self.tuple_misses,
                    row_hits=self.row_hits, row_misses=self.row_misses,
                    memo_rebuilds=self.memo_rebuilds, memo_bound_discards=self.memo_bound_discards,
                    owner_shortcut_hits=self.owner_shortcut_hits,
                    owner_shortcut_rejections=self.owner_shortcut_rejections,
                    witnesses=self.witnesses, witness_ineligible=self.witness_ineligible,
                    held_witness=None if witness is None else witness.size(),
                    estimated_retained_bytes=retained, closed=self._closed)

    def close(self):
        self._tuple_owns, self._tuple_pins, self._row_owns = {}, {}, {}
        self._last_owner_call = None
        self._closed = True


def payload_storage_bytes(*values):
    """Stateless exact API; production substitution uses an explicit session."""
    session = AccountingSession()
    try:
        return session(*values)
    finally:
        session.close()

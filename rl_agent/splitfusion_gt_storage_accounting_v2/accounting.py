"""Exact V1 accounting with bounded summaries of immutable ``_PairHistory`` objects.

The logical byte definition is unchanged. Snapshot summaries are keyed by the
exact prior-seen descendant intersection, so a context-partial result can never
replace or satisfy a full-snapshot lookup. Reuse also requires unchanged CPython
dict mutation tags, immutable PairHistory identities and strongly pinned signed
objects. If mutation tags are unavailable, snapshot and owner reuse fail closed.
Entries are identity guarded, LRU bounded, and owned by one explicit session.
"""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import sys
from types import FunctionType
import weakref

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1.conflicts import _PairHistory
from rl_agent.splitfusion_numeric_gt_integration_v2.ownership import OwnershipError
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as V1

try:
    from _testcapi import dict_get_version as _dict_version
except ImportError:  # Fail closed on interpreters without CPython's mutation tag.
    _dict_version = None

VERSION = "splitfusion_gt_storage_accounting_v2.bounded_pair_history_summary.v2"
DEFAULT_MAX_ENTRIES = 4096
_CONTAINERS = (list, tuple, set, frozenset)
_SCALAR = V1._SCALAR


@dataclass(frozen=True, slots=True)
class _Entry:
    reference: object
    signature: tuple
    descendants: frozenset
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _SnapshotEntry:
    snapshot: dict
    signature: tuple
    pins: tuple
    descendants: frozenset
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _SnapshotState:
    signature: tuple
    descendants: frozenset


@dataclass(frozen=True, slots=True)
class _SnapshotGuard:
    signature: tuple
    pins: tuple


def _signature(value):
    """Return the O(1) immutable-container identity signature."""
    if type(value) is not _PairHistory or type(value.overlap_count) is not int:
        return None
    sequences = (value.iou, value.ranges, value.near)
    if any(type(sequence) is not tuple for sequence in sequences):
        return None
    if (type(value.best) is not tuple or len(value.best) != 2
            or type(value.best[0]) is not float or type(value.best[1]) is not str):
        return None
    return tuple((id(sequence), len(sequence)) for sequence in sequences) + (
        value.overlap_count, id(value.best), len(value.best))


def _valid_pair_contents(value):
    """One-time validation before a new identity signature becomes reusable."""
    return all(type(item) is float for sequence in (value.iou, value.ranges, value.near)
               for item in sequence)


def _summary(value):
    """V1's exact dataclass/tuple/scalar traversal, plus visited descendants."""
    seen = set()

    def size(item):
        if item is None:
            return 1
        if isinstance(item, str):
            return 8 + len(item.encode("utf-8"))
        if isinstance(item, (bool, int, float, np.generic)):
            return 16
        ident = id(item)
        if ident in seen:
            return 0
        seen.add(ident)
        if type(item) is tuple:
            return 16 + sum(8 + size(child) for child in item)
        raise TypeError("non-immutable pair-history descendant")

    return 16 + sum(size(getattr(value, field.name)) for field in fields(value)), frozenset(seen)


_SNAPSHOT_KEYS = frozenset(("templates", "pairs", "decisions", "frame_count", "row_count", "previous_frame",
                            "previous_timestamp", "first_timestamp", "episode", "seen_samples", "chain"))
_SNAPSHOT_SCALARS = ("frame_count", "row_count", "previous_frame", "previous_timestamp",
                     "first_timestamp", "episode", "chain")


def _snapshot_shape(value):
    return type(value) is dict and set(value) == _SNAPSHOT_KEYS


def _snapshot_token(value):
    """Return the cheap exact mutation/type/immutable-identity token."""
    if _dict_version is None or not _snapshot_shape(value):
        return None
    try:
        templates, decisions, pairs, samples = (value[name] for name in
                                                ("templates", "decisions", "pairs", "seen_samples"))
        if any(type(mapping) is not dict for mapping in (templates, decisions, pairs)) or type(samples) is not set:
            return None
        def pair_key(key):
            if (type(key) is not tuple or len(key) != 2
                    or any(type(part) is not str for part in key)):
                return None
            return id(key)

        mapping_states = []
        for mapping in (templates, decisions):
            rows = []
            for key, row in mapping.items():
                key_state = pair_key(key)
                if key_state is None or type(row) is not dict:
                    return None
                rows.append((key_state, id(row), _dict_version(row)))
            mapping_states.append((id(mapping), _dict_version(mapping), tuple(rows)))

        pair_states = []
        for key, history in pairs.items():
            key_state, history_state = pair_key(key), _signature(history)
            if key_state is None or history_state is None:
                return None
            pair_states.append((key_state, id(history), history_state))

        return (_dict_version(value), tuple(mapping_states),
                (id(pairs), _dict_version(pairs), tuple(pair_states)),
                (id(samples), frozenset(map(id, samples)), frozenset(map(type, samples))),
                tuple((id(value[name]), type(value[name])) for name in _SNAPSHOT_SCALARS))
    except (AttributeError, TypeError):
        return None


def _snapshot_state(value, signature=None):
    """Add the descendant topology once, only when a new context is missed."""
    signature = _snapshot_token(value) if signature is None else signature
    if signature is None:
        return None
    descendants = {id(value[name]) for name in ("templates", "decisions", "pairs", "seen_samples")}
    for name in ("templates", "decisions"):
        for key, row in value[name].items():
            descendants.update((id(key), id(row)))
    for key, history in value["pairs"].items():
        descendants.update((id(key), id(history), id(history.iou), id(history.ranges),
                            id(history.near), id(history.best)))
    return _SnapshotState(signature, frozenset(descendants))


def _freeze_snapshot(value):
    """Validate once and pin every identity represented by the signature."""
    if (any(type(key) is not tuple or len(key) != 2 or any(type(part) is not str for part in key)
            for name in ("templates", "decisions", "pairs") for key in value[name])
            or any(type(row) is not dict or any(type(item) not in _SCALAR for pair in row.items() for item in pair)
                   for name in ("templates", "decisions") for row in value[name].values())
            or any(_signature(history) is None or not _valid_pair_contents(history)
                   for history in value["pairs"].values())
            or type(value["seen_samples"]) is not set
            or any(type(sample) is not str for sample in value["seen_samples"])
            or any(type(value[name]) not in _SCALAR for name in _SNAPSHOT_SCALARS)):
        return None
    pins = [value[name] for name in ("templates", "decisions", "pairs", "seen_samples")]
    for name in ("templates", "decisions"):
        for key, row in value[name].items():
            pins.extend((key, row)); pins.extend(row.keys()); pins.extend(row.values())
    for key, history in value["pairs"].items():
        pins.extend((key, history, history.iou, history.ranges, history.near, history.best))
    pins.extend(value["seen_samples"])
    pins.extend(value[name] for name in _SNAPSHOT_SCALARS)
    return tuple(pins)


_OWNER_INELIGIBLE = object()


def _owner_snapshot_state(values):
    """Locate the rollback snapshot and its cheap exact token."""
    if len(values) != 3:
        return _OWNER_INELIGIBLE
    staged = values[1]
    restore = staged.get("_restore") if type(staged) is dict else getattr(staged, "_restore", None)
    if type(restore) is not dict or "conflict" not in restore:
        return _OWNER_INELIGIBLE
    snapshot = restore["conflict"]
    signature = _snapshot_token(snapshot)
    if signature is None:
        return _OWNER_INELIGIBLE
    return snapshot, signature


class AccountingSession:
    """Process-local callable installed only in the GT preparation worker."""

    def __init__(self, max_entries=DEFAULT_MAX_ENTRIES, max_snapshots=8):
        if (type(max_entries) is not int or max_entries < 1
                or type(max_snapshots) is not int or max_snapshots < 1):
            raise ValueError("positive bounded accounting-summary capacity required")
        self.max_entries, self.max_snapshots = max_entries, max_snapshots
        self._entries = OrderedDict()
        self._snapshots = OrderedDict()
        self._closed = False
        self.calls = self.hits = self.misses = self.invalidations = 0
        self.alias_fallbacks = self.evictions = self.expired = 0
        self.snapshot_hits = self.snapshot_misses = self.snapshot_invalidations = 0
        self.snapshot_alias_fallbacks = self.snapshot_evictions = 0
        self.owner_shortcut_hits = 0
        self._last_owner_call = None

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

    @staticmethod
    def _scalar_size(value):
        if value is None:
            return 1
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return 8 + (len(value.encode("utf-8")) if isinstance(value, str) else
                        value.nbytes if isinstance(value, memoryview) else len(value))
        if isinstance(value, (bool, int, float, np.generic)):
            return 16
        return None

    def _expire(self, ident, reference):
        entry = self._entries.get(ident)
        if entry is not None and entry.reference is reference:
            self._entries.pop(ident, None)
            self.expired += 1

    def _entry(self, value, signature):
        ident = id(value)
        entry = self._entries.get(ident)
        if entry is not None:
            if entry.reference() is value and entry.signature == signature:
                self._entries.move_to_end(ident)
                return entry, True
            self._entries.pop(ident, None)
            self.invalidations += 1
        self.misses += 1
        if not _valid_pair_contents(value):
            return None, False
        logical_bytes, descendants = _summary(value)
        owner = weakref.ref(self)
        def expired(reference, key=ident, owner=owner):
            session = owner()
            if session is not None:
                session._expire(key, reference)
        reference = weakref.ref(value, expired)
        entry = _Entry(reference, signature, descendants, logical_bytes)
        self._entries[ident] = entry
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        return entry, False

    def __call__(self, *values):
        if self._closed:
            raise RuntimeError("accounting summary session is closed")
        self.calls += 1
        caller = self._owner_caller()
        owner_binding = _owner_snapshot_state(values) if caller is not None else _OWNER_INELIGIBLE
        if owner_binding is _OWNER_INELIGIBLE:
            owner_snapshot, owner_signature = None, None
        else:
            owner_snapshot, owner_signature = owner_binding
        previous = self._last_owner_call
        if (caller in ("bind_plan", "set_ready") and previous is not None
                and (previous[0], caller) in (("stage_products", "bind_plan"),
                                              ("bind_plan", "set_ready"),
                                              ("stage_products", "set_ready"))
                and len(values) == len(previous[1]) == 3
                and values[1] is previous[1][1] and values[2] is previous[1][2]
                and owner_signature is not None and isinstance(previous[3], _SnapshotGuard)
                and owner_signature == previous[3].signature):
            prior_plan, current_plan = previous[1][0], values[0]
            if current_plan is prior_plan:
                result = previous[2]
            else:
                prior_size, current_size = self._scalar_size(prior_plan), self._scalar_size(current_plan)
                result = None if prior_size is None or current_size is None else previous[2] - prior_size + current_size
            if result is not None:
                self.owner_shortcut_hits += 1
                self._last_owner_call = (caller, values, result, previous[3])
                return result
        seen = set()
        owner_guard_seen = None

        def size(value):
            nonlocal owner_guard_seen
            if value is None:
                return 1
            if isinstance(value, (str, bytes, bytearray, memoryview)):
                return 8 + (len(value.encode("utf-8")) if isinstance(value, str) else
                            value.nbytes if isinstance(value, memoryview) else len(value))
            if isinstance(value, (bool, int, float, np.generic)):
                return 16
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise OwnershipError("object ndarray cannot be retained")
                return 32 + value.nbytes
            if isinstance(value, Mapping):
                snapshot_shape = _snapshot_shape(value)
                signature = (owner_signature if value is owner_snapshot else _snapshot_token(value)) \
                    if snapshot_shape else None
                if snapshot_shape:
                    entries = [(key, entry) for key, entry in self._snapshots.items()
                               if key[0] == ident and entry.snapshot is value]
                    matching = next(((key, entry) for key, entry in entries
                                     if signature is not None and entry.signature == signature), None)
                    if matching is None:
                        for key, _entry_value in entries:
                            self._snapshots.pop(key, None)
                        self.snapshot_invalidations += len(entries)
                        snapshot_state = _snapshot_state(value, signature) if signature is not None else None
                        descendants = snapshot_state.descendants if snapshot_state is not None else frozenset()
                        validation_pins = None
                    else:
                        descendants = matching[1].descendants
                        validation_pins = matching[1].pins
                    overlap = frozenset(descendants.intersection(seen)) if signature is not None else None
                    cache_key = (ident, overlap)
                    cached = self._snapshots.get(cache_key)
                    if (signature is not None and cached is not None and cached.snapshot is value
                            and cached.signature == signature and cached.descendants == descendants):
                        self._snapshots.move_to_end(cache_key)
                        seen.update(cached.descendants)
                        self.snapshot_hits += 1
                        if value is owner_snapshot:
                            owner_guard_seen = _SnapshotGuard(signature, cached.pins)
                        return cached.logical_bytes
                    self.snapshot_misses += 1
                    if overlap:
                        self.snapshot_alias_fallbacks += 1
                    result = 16 + sum(8 + size(key) + size(item) for key, item in value.items())
                    after = _snapshot_token(value) if signature is not None else None
                    pins = (validation_pins if signature is not None and after == signature
                            and validation_pins is not None else _freeze_snapshot(value)
                            if signature is not None and after == signature else None)
                    if pins is not None:
                        self._snapshots[cache_key] = _SnapshotEntry(
                            value, signature, pins, descendants, result)
                        if value is owner_snapshot:
                            owner_guard_seen = _SnapshotGuard(signature, pins)
                        if len(self._snapshots) > self.max_snapshots:
                            self._snapshots.popitem(last=False)
                            self.snapshot_evictions += 1
                    return result
                if type(value) is dict:
                    keys, items = list(value), list(value.values())
                    key_kinds, item_kinds = set(map(type, keys)), set(map(type, items))
                    if key_kinds <= _SCALAR and item_kinds <= _SCALAR:
                        try:
                            return (16 + 8 * len(keys) + V1._flat(keys, key_kinds)
                                    + V1._flat(items, item_kinds))
                        except Exception:
                            pass
                return 16 + sum(8 + size(key) + size(item) for key, item in value.items())
            if isinstance(value, (list, tuple, set, frozenset)):
                if type(value) in _CONTAINERS:
                    elements = list(value)
                    kinds = set(map(type, elements))
                    if kinds <= _SCALAR:
                        try:
                            return 16 + 8 * len(elements) + V1._flat(elements, kinds)
                        except Exception:
                            pass
                return 16 + sum(8 + size(item) for item in value)
            if is_dataclass(value):
                signature = _signature(value)
                if signature is not None:
                    entry, reused = self._entry(value, signature)
                    if entry is not None and entry.descendants.isdisjoint(seen):
                        seen.update(entry.descendants)
                        self.hits += int(reused)
                        return entry.logical_bytes
                    self.alias_fallbacks += 1
                return 16 + sum(size(getattr(value, field.name)) for field in fields(value))
            if isinstance(value, FunctionType):
                return 16 + sum(size(cell.cell_contents) for cell in (value.__closure__ or ()))
            raise OwnershipError("unsupported retained payload type: " + type(value).__name__)

        result = sum(size(value) for value in values)
        owner_guard = (owner_guard_seen if caller in ("stage_products", "bind_plan")
                       and owner_guard_seen is not None else _OWNER_INELIGIBLE)
        self._last_owner_call = ((caller, values, result, owner_guard)
                                 if caller is not None else None)
        return result

    def snapshot(self):
        retained = (sys.getsizeof(self._entries) + sys.getsizeof(self._snapshots)
                    + sum(sys.getsizeof(key) + sys.getsizeof(entry) + sys.getsizeof(entry.descendants)
                          + sum(sys.getsizeof(ident) for ident in entry.descendants)
                          for key, entry in self._entries.items())
                    + sum(sys.getsizeof(key) + sys.getsizeof(entry) + sys.getsizeof(entry.descendants)
                          + sum(sys.getsizeof(ident) for ident in entry.descendants)
                          for key, entry in self._snapshots.items()))
        retained_logical_upper_bound = sum(entry.logical_bytes for entry in self._snapshots.values())
        return dict(version=VERSION, max_entries=self.max_entries, entries=len(self._entries),
                    max_snapshots=self.max_snapshots, snapshots=len(self._snapshots), calls=self.calls,
                    hits=self.hits, misses=self.misses, invalidations=self.invalidations,
                    alias_fallbacks=self.alias_fallbacks, evictions=self.evictions, expired=self.expired,
                    snapshot_hits=self.snapshot_hits, snapshot_misses=self.snapshot_misses,
                    snapshot_invalidations=self.snapshot_invalidations,
                    snapshot_alias_fallbacks=self.snapshot_alias_fallbacks,
                    snapshot_evictions=self.snapshot_evictions,
                    owner_shortcut_hits=self.owner_shortcut_hits,
                    estimated_retained_bytes=retained,
                    retained_snapshot_logical_bytes_upper_bound=retained_logical_upper_bound,
                    closed=self._closed)

    def close(self):
        self._entries.clear()
        self._snapshots.clear()
        self._last_owner_call = None
        self._closed = True


def payload_storage_bytes(*values):
    """Stateless exact API; production substitution uses an explicit session."""
    session = AccountingSession(max_entries=1)
    try:
        return session(*values)
    finally:
        session.close()

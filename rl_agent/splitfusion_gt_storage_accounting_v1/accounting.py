"""Exact candidate for ``splitfusion_numeric_gt_integration_v2.ownership.payload_storage_bytes``.

The original walks every element of every container in Python.  In the GT prep worker the dominant containers are the
``_PairHistory`` observation tuples of the full causal conflict history, held by ``Staged._restore["conflict"]``; the
walk therefore grows with every static-pair observation ever accepted (V8 diagnosis,
``experiments/splitfusion_m3_gt_prep_growth_diag_v1/REPORT.md``).

The candidate is a line-for-line copy of the original with two changes, each inside one existing branch and each
entered only AFTER the container's own identity has been checked and registered in ``seen`` exactly as before:

* ``Mapping`` branch: if ``type(value) is dict`` and every key and every value has an exact type in ``_SCALAR``
  (whitelisted immutable numeric scalars, ``str``, ``None``), the closed form
  ``16 + 8 * len + flat(keys) + flat(values)`` is returned;
* ``list/tuple/set/frozenset`` branch: if the exact type is one of those four and every element's exact type is in
  ``_SCALAR``, ``16 + 8 * len + flat(elements)`` is returned;
* otherwise -- or if the closed form raises anything (e.g. a lone surrogate in a ``str``) -- the original loop runs,
  which then raises the original exception itself.

``flat`` is the sum of the original scalar sizes: 16 per numeric scalar, 1 per ``None``, ``8 + len(utf-8)`` per
``str`` (UTF-8 is per code point, so the encoded length of the concatenation equals the sum of the encoded lengths, and
the encoding raises if and only if some element contains a lone surrogate).

Why this is exact (and why no cache is needed):
* For an element whose exact type is in ``_SCALAR`` the original ``size`` returns a constant (numeric 16, ``None`` 1)
  or ``8 + len(utf-8)`` (``str``) WITHOUT reading or writing ``seen``.  Such elements therefore contribute a value
  that depends only on their own immutable contents -- never on traversal order, on other top-level arguments or on
  what was visited before -- so their sum has a closed form.
* Containers -- including shared child tuples, the shared empty tuple, reused ``_PairHistory.best`` tuples and dicts
  referenced twice -- are still deduplicated by the unchanged ``seen`` logic, because each fast path runs only inside
  the branch the original would take, after that container's own ``id`` check/registration.  A container holding any
  non-scalar takes the original loop, so nested containers are visited exactly as before.
* Exact-type tests (``type(x) is ...``) are used, never ``isinstance``, so subclasses with custom ``__iter__``,
  ``__len__``, ``items``, ``__class__`` or ``encode`` take the original path.  The whitelisted numeric types are static
  C types whose instances cannot rebind ``__class__``; ``_check_whitelist`` re-proves at import that each takes the
  original scalar branch.
* Nothing is retained between calls (no cache): eviction, garbage collection and ``id`` reuse cannot affect a result.
  Python-level recursion per scalar (per observation, per row field) is replaced by C-level ``map(type, ...)`` scans;
  the remaining Python work is per container (per pair / per row), not per scalar.
* Every rejection rule (object ndarray, unsupported type) is unchanged: the fast paths accept only scalars, which the
  original never rejects.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import FunctionType

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v2.ownership import OwnershipError

VERSION = "splitfusion_gt_storage_accounting_v1.flat_scalar_containers.v2"
_CONTAINERS = (list, tuple, set, frozenset)
_NUMERIC = frozenset((bool, int, float, np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                      np.uint32, np.uint64, np.float16, np.float32, np.float64, np.longdouble))
_NONE = type(None)
_SCALAR = _NUMERIC | {str, _NONE}


def _check_whitelist():
    for kind in _NUMERIC:
        if (not isinstance(kind, type) or issubclass(kind, (str, bytes, bytearray, memoryview))
                or not issubclass(kind, (bool, int, float, np.generic)) or kind.__flags__ & (1 << 9)):
            # 1 << 9 = Py_TPFLAGS_HEAPTYPE: only static (C) types, whose instances cannot rebind __class__
            raise RuntimeError(f"accounting whitelist type is not a static numeric scalar: {kind!r}")


_check_whitelist()


def _flat(items, kinds):
    """Sum of the original scalar sizes of ``items`` (a list) whose exact types are ``kinds`` (a subset of _SCALAR)."""
    if kinds <= _NUMERIC:
        return 16 * len(items)
    strings = [x for x in items if type(x) is str]
    nones = sum(1 for x in items if x is None) if _NONE in kinds else 0
    return (16 * (len(items) - len(strings) - nones) + nones + 8 * len(strings)
            + len("".join(strings).encode("utf-8")))


def payload_storage_bytes(*values):
    """Conservative logical storage: buffers, UTF-8, scalars and framing.

    Shared containers are counted once, ndarray views conservatively count their
    visible bytes. Includes Staged rollback deltas and closure-owned class masks.
    It deliberately makes no claim about allocator/object overhead or peak RSS.
    """
    seen = set()

    def size(value):
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
            if type(value) is dict:                                          # --- change 1 of 2
                keys, items = list(value), list(value.values())
                key_kinds, item_kinds = set(map(type, keys)), set(map(type, items))
                if key_kinds <= _SCALAR and item_kinds <= _SCALAR:
                    try:
                        return 16 + 8 * len(keys) + _flat(keys, key_kinds) + _flat(items, item_kinds)
                    except Exception:                                    # the original loop raises it exactly
                        pass                                             # --- end of change 1
            return 16 + sum(8 + size(k) + size(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set, frozenset)):
            if type(value) in _CONTAINERS:                                   # --- change 2 of 2
                elements = list(value)
                kinds = set(map(type, elements))
                if kinds <= _SCALAR:
                    try:
                        return 16 + 8 * len(elements) + _flat(elements, kinds)
                    except Exception:                                    # the original loop raises it exactly
                        pass                                             # --- end of change 2
            return 16 + sum(8 + size(v) for v in value)
        if is_dataclass(value):
            return 16 + sum(size(getattr(value, f.name)) for f in fields(value))
        if isinstance(value, FunctionType):
            # Staged.same_class owns a dict of image masks and semantic ndarray.
            return 16 + sum(size(c.cell_contents) for c in (value.__closure__ or ()))
        raise OwnershipError("unsupported retained payload type: " + type(value).__name__)

    return sum(size(v) for v in values)

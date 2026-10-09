"""Exact CPU-only source preparation in private, per-producer namespaces.

No geometry rule, evidence field, conflict state, source pin or input validator
is removed. Only equivalent canonical conversion and fixed-grid nearest mask
sampling are accelerated. Original modules are never mutated or shared as
per-frame IO namespaces.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
import types

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as C
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A


BASE_PRODUCER = S.LiveCompleteTargetProducer


def json_value(value):
    """Exactly A._json_value, avoiding typing.Mapping checks on primitives."""
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return value
    if kind is dict or isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=A.canonical)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def _function(original, namespace):
    result = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    result.__annotations__ = original.__annotations__
    return result


@lru_cache(maxsize=1)
def _fixed_grid_proof():
    """Exhaustively verify every output pixel's original source index once."""
    h, w, mh, mw = 720, 1280, 432, 768
    ys = np.clip(np.floor(np.arange(mh)*(h/float(mh))).astype(np.int64), 0,h-1)
    xs = np.clip(np.floor(np.arange(mw)*(w/float(mw))).astype(np.int64), 0,w-1)
    # Native-to-model scale is exactly 5/3 in each dimension. The original
    # float-floor indices must equal residues0,1,3 in each5-pixel block. Prove
    # every axis index, hence their full Cartesian product, with no rounding
    # assumption. Each of the nine views below is disjoint; integer nonzero
    # counts are therefore additive and order-independent.
    yy=np.sort(np.concatenate([np.arange(offset,h,5) for offset in (0,1,3)]))
    xx=np.sort(np.concatenate([np.arange(offset,w,5) for offset in (0,1,3)]))
    if not np.array_equal(ys,yy) or not np.array_equal(xs,xx):
        raise S.SourceGeometryError("strided nearest mapping differs from original fixed grid")
    return True


def _nearest_model_count(mask, model_width, model_height):
    # The proof covers this sole production geometry. All other dimensions or
    # dtypes retain the exact original implementation rather than extrapolate.
    if (mask.shape == (720,1280) and model_width == 768 and model_height == 432
            and mask.dtype in (np.dtype(bool), np.dtype(np.uint8))):
        _fixed_grid_proof()
        return sum(int(np.count_nonzero(mask[y::5,x::5]))
                   for y in (0,1,3) for x in (0,1,3))
    return S._original()[3]._nearest_model_count(mask, model_width, model_height)


def _adapter_namespace():
    namespace = dict(vars(A))
    namespace["_json_value"] = json_value
    for name in ("input_content_sha256", "_validate"):
        namespace[name] = _function(getattr(A,name), namespace)
    return types.SimpleNamespace(**namespace)


def _conflict_namespace(adapter):
    namespace = dict(vars(C), A=adapter)
    for name in ("_sha", "catalogue_content_sha256", "derive_live_quarantine"):
        namespace[name] = _function(getattr(C,name), namespace)
    methods = {name: _function(getattr(C.IncrementalConflictState,name), namespace)
               for name in ("__init__", "derive")}
    namespace["IncrementalConflictState"] = type("PrivateExactConflictState",
        (C.IncrementalConflictState,), methods)
    return types.SimpleNamespace(**namespace)


def _original_private():
    objects, semantic, static0, visibility0, paint = S._original()
    visibility = dict(vars(visibility0))
    for name, value in tuple(visibility.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(visibility0):
            visibility[name] = _function(value, visibility)
    visibility["_nearest_model_count"] = _nearest_model_count
    private_visibility = types.SimpleNamespace(**visibility)
    static = dict(static0)
    for name, value in tuple(static.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is static0:
            static[name] = _function(value, static)
    static["visibility_from_corners"] = private_visibility.visibility_from_corners
    # S._static_rows clones functions from this private static namespace again
    # for each frame before installing image/radar IO proxies.
    return objects, semantic, static, private_visibility, paint


def _producer_methods():
    adapter = _adapter_namespace()
    namespace = dict(vars(S), A=adapter, conflicts=_conflict_namespace(adapter),
                     _original=_original_private)
    namespace["_sha"] = _function(S._sha, namespace)
    return {name: _function(getattr(BASE_PRODUCER,name), namespace)
            for name in ("__init__", "build")}


_METHODS = _producer_methods()


class OptimizedSourceProducer(BASE_PRODUCER):
    __init__ = _METHODS["__init__"]
    build = _METHODS["build"]


def prewarm():
    """Safe after import, before any live action; no simulator/server needed."""
    sources = S.verify_sources()
    S._original()
    C._original_functions()
    _fixed_grid_proof()
    return {"source_pins": sources, "fixed_grid_proved": True,
            "state_or_history_created": False}


@contextmanager
def install():
    """One fresh child process only; compatible with the GT timing wrapper."""
    previous = S.LiveCompleteTargetProducer
    if previous is not BASE_PRODUCER:
        raise RuntimeError("unexpected source producer seam; install before timing wrapper")
    prewarm()
    S.LiveCompleteTargetProducer = OptimizedSourceProducer
    try:
        yield
    finally:
        S.LiveCompleteTargetProducer = previous

"""Exact GT-only CPU replacements in private function namespaces.

No cached frame values, validation results or hashes. Full uint8 label sets
are computed by counting every pixel rather than sorting it; all other NumPy
signatures retain the original operation. All file IO/fsync/READY code remains
the original code. This scope must be inside the already-qualified v1 scope.
It is not a live entry point or authority to run a route.
"""
from contextlib import contextmanager
import threading
import types

import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_gt_optimization_v1 import target_builder as V1T
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A

_LOCK = threading.Lock()
_BASE_ENTRYPOINTS = {
    (E, "prepare_target"): E.prepare_target,
    (E, "restore_prepared_target"): E.restore_prepared_target,
    (T, "write_bundle"): T.write_bundle,
    (G, "write_source_archive"): G.write_source_archive,
}


class ExactArrayOperations:
    """No array retained; unexpected signatures and subclasses fall through."""
    def __getattr__(self, name):
        return getattr(np, name)

    def unique(self, value, *args, **kwargs):
        if type(value) is np.ndarray and value.dtype == np.uint8 and not args and not kwargs:
            counts = np.bincount(value.ravel(order="C"), minlength=256)
            return np.flatnonzero(counts).astype(np.uint8)
        return np.unique(value, *args, **kwargs)


def clone_functions(module, **overrides):
    """Preserve original classes/exceptions and transitive function bindings."""
    original = vars(module)
    namespace = dict(original)
    for name, value in original.items():
        if isinstance(value, types.FunctionType) and value.__globals__ is original:
            namespace[name] = V1S._function(value, namespace)
    namespace.update(overrides)
    return namespace


def private_functions(*, fast_labels=True, fast_metadata=True):
    array_ops = ExactArrayOperations() if fast_labels else np
    adapter = clone_functions(A, np=array_ops)
    if fast_metadata:
        adapter["_json_value"] = V1S.json_value
    private_adapter = types.SimpleNamespace(**adapter)
    target = clone_functions(V1T, A=private_adapter)
    # Uses the v1 memory-backed temporary builder with all original validators.
    private_adapter.build_complete_v010 = target["build_complete_v010"]
    evaluation = clone_functions(E, targets=private_adapter, np=array_ops)
    transport = clone_functions(T, targets=private_adapter)
    bridge = clone_functions(G, A=private_adapter)
    # V1 already has the fast converter in its private source adapter. Retain
    # it when measuring label-only ablation, but replace its NumPy dispatch.
    source_adapter = V1S._adapter_namespace()
    source_globals = dict(V1S.OptimizedSourceProducer.build.__globals__)
    if fast_labels:
        source_ns = dict(vars(source_adapter), np=array_ops)
        for name in ("_validate", "input_content_sha256"):
            source_ns[name] = V1S._function(source_ns[name], source_ns)
        source_adapter = types.SimpleNamespace(**source_ns)
    source_globals["A"] = source_adapter
    build = V1S._function(V1S.OptimizedSourceProducer.build, source_globals)
    producer = type("ExactCpuSourceProducerV2", (V1S.OptimizedSourceProducer,), {"build": build})
    return dict(producer=producer, prepare=evaluation["prepare_target"],
                restore=evaluation["restore_prepared_target"],
                write=transport["write_bundle"], archive=bridge["write_source_archive"])


@contextmanager
def install(*, fast_labels=True, fast_metadata=True):
    """Scoped UE functions only; restore even on failure, never touch NumPy."""
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("v2 GT scope cannot nest or install concurrently")
    slots = ((S, "LiveCompleteTargetProducer"), (E, "prepare_target"),
             (E, "restore_prepared_target"), (T, "write_bundle"), (G, "write_source_archive"))
    saved = [getattr(module, name) for module, name in slots]
    try:
        if S.LiveCompleteTargetProducer is not V1S.OptimizedSourceProducer:
            raise RuntimeError("v2 additions require the exact retained v1 GT scope")
        if any(getattr(module, name) is not original
               for (module, name), original in _BASE_ENTRYPOINTS.items()):
            raise RuntimeError("v2 additions require original GT entrypoints; install before timing wrappers")
        functions = private_functions(fast_labels=fast_labels, fast_metadata=fast_metadata)
        for (module, name), key in zip(slots, ("producer", "prepare", "restore", "write", "archive")):
            setattr(module, name, functions[key])
        yield {"fast_labels": fast_labels, "fast_metadata": fast_metadata,
               "all_pixel_validation": True, "frame_value_cache": False,
               "fsync_unchanged": True, "deferred_evidence": False,
               "new_threads": 0, "new_gpu_work": False, "live_qualified": False}
    finally:
        for (module, name), value in zip(slots, saved):
            setattr(module, name, value)
        _LOCK.release()

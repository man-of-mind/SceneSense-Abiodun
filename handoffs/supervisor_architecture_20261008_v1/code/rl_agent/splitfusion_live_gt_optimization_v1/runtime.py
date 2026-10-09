"""Scoped GT-only integration, installed outside the retained UE/timing scopes.

This is deliberately not a live CLI. A future launch must prospectively bind
this source inventory alongside the existing v3 runtime before calling it.
No existing sealed registration is silently made to execute different code.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

PACKAGE = Path(__file__).resolve().parent
_LOCK = threading.Lock()
_SETTINGS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")


def source_inventory():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE.glob("*.py")) if not p.name.startswith("test_")}


def verify_sources(expected):
    if not isinstance(expected, dict) or not expected or expected != source_inventory():
        raise ValueError("GT optimization source inventory differs from prospective binding")


def settings():
    value = {"environment": {name: os.environ.get(name) for name in _SETTINGS},
             "calling_thread_affinity": sorted(os.sched_getaffinity(0))}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        value["torch_threads"] = torch.get_num_threads()
        value["torch_interop_threads"] = torch.get_num_interop_threads()
        value["cuda_initialized"] = torch.cuda.is_initialized()
    if "cv2" in sys.modules:
        value["opencv_threads"] = sys.modules["cv2"].getNumThreads()
    return value


@contextmanager
def install_cpu_gt_optimizations(*, expected_sources):
    """Prewarm code only, then replace two exact GT implementations.

    Install BEFORE repair_v2.install_gt_timing and the v3 UE composition. The
    existing owned GT worker must be drained by that composition before exit.
    All numerical validation, source ownership, stage timing, target transport,
    READY/fsync, edge evaluation and postflight functions remain unchanged.
    """
    verify_sources(expected_sources)
    owned_sources = dict(expected_sources)
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("GT optimization scope cannot be nested or installed concurrently")
    try:
        from . import source_geometry as source
        from . import target_builder as target
        before = settings()
        begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        prewarm = {"source": source.prewarm(), "target": target.prewarm()}
        state = {"schema":"splitfusion.cpu_gt_optimization.scope.v1", "prewarm":prewarm,
                 "prewarm_wall_ns":time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)-begin,
                 "new_python_threads":0, "new_queues":0, "new_gpu_work":False,
                 "forensic_writes_deferred":False, "live_qualified":False,
                 "sources":dict(owned_sources), "settings_before":before}
        if settings() != before:
            raise RuntimeError("GT prewarm changed a retained process setting")
        failed = False
        try:
            with ExitStack() as stack:
                stack.enter_context(source.install())
                stack.enter_context(target.install_target_builder())
                yield state
        except BaseException:
            failed = True
            raise
        finally:
            # Restore seams first, record failures even when the caller failed,
            # and never replace the caller's exception with an audit exception.
            state["settings_after"] = settings()
            state["retained_settings_unchanged"] = state["settings_after"] == before
            state["source_integrity_error"] = None
            try:
                verify_sources(owned_sources)
            except Exception as exc:
                state["source_integrity_error"] = str(exc)
            if not failed:
                if state["source_integrity_error"] is not None:
                    raise RuntimeError(state["source_integrity_error"])
                if not state["retained_settings_unchanged"]:
                    raise RuntimeError("retained process settings changed during GT optimization scope")
    finally:
        _LOCK.release()

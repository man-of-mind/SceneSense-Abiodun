"""Truthful mixed-TID context around the unchanged dedicated GT request loop."""
import os
import sys
from . import contract as C


def main():
    if any(name in sys.modules for name in ("torch", "numpy", "cv2")):
        raise RuntimeError("main-thread placement bootstrap must precede numerical imports")
    os.environ.update(C.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W, snapshot as S
    from .engine import Engine
    from .placement import scheduling
    expected = C.sources()
    old_engine, old_context, old_unpack = W.Engine, W.verify_child_context, S.unpack_snapshot
    owned = []
    numerical_context = None

    class Observed(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            owned.append(self)

    def context(original_expected):
        nonlocal numerical_context
        import torch
        import cv2
        W.require(len(owned) == 1, "one main-thread GT placement engine required")
        engine = owned[0]
        placement = engine.placement
        if not placement.closed:
            placement.end_request("FAILED" if engine.failed else "OK")
        W.require(C.sources() == expected, "registered main-thread placement sources changed")
        W.require(original_expected == placement.original_scheduling,
                  "original parent scheduling binding differs")
        if not placement.closed:
            rows = placement.verify()
        else:
            rows = placement.restored_threads
            placement._check_threads(rows, original_expected)
        W.require(not torch.cuda.is_initialized(), "CPU worker initialized CUDA")
        numeric = dict(cuda_initialized=False, torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(), opencv_threads=cv2.getNumThreads(),
            environment={name: os.environ.get(name) for name in C.ENVIRONMENT})
        if numerical_context is None:
            numerical_context = numeric
        W.require(numeric == numerical_context, "retained numerical settings changed")
        actual = scheduling()
        W.require(actual == (original_expected if placement.closed else placement.active_scheduling),
                  "main-thread effective scheduling differs")
        # W.verify_child_context assumes uniform masks. Report the authentic
        # mixed contexts instead, retaining all its numerical/environment and
        # scheduler/nice checks through this explicitly bound validator.
        threads = {tid: {key: row[key] for key in ("affinity", "scheduler", "nice")}
                   for tid, row in rows.items()}
        result = dict(numeric, pid=os.getpid(), scheduling=actual, threads=threads,
            runtime_gc=engine.gc_policy.snapshot(include_events=False),
            runtime_placement=placement.snapshot(), original_parent_scheduling=original_expected,
            thread_context_validator="explicit_main_singleton_helpers_original")
        return C.validate_context(dict(result, bootstrap_sources=expected), expected)

    def unpack(payload):
        W.require(len(owned) == 1, "snapshot needs one main-thread GT owner")
        placement = owned[0].placement
        placement.begin_request()
        try:
            result = old_unpack(payload)
            placement.identify_request(result[0]["identity"])
            return result
        except BaseException:
            placement.end_request("SNAPSHOT_FAILED")
            raise

    W.Engine, W.verify_child_context, S.unpack_snapshot = Observed, context, unpack
    try:
        return W.main()
    finally:
        W.Engine, W.verify_child_context, S.unpack_snapshot = old_engine, old_context, old_unpack
        for engine in owned:
            if engine.placement is not None and not engine.placement.closed:
                engine.placement.restore()
            if engine.gc_policy is not None and not engine.gc_policy.closed:
                engine.gc_policy.close()


if __name__ == "__main__":
    raise SystemExit(main())

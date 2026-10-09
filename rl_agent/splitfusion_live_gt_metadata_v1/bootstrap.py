"""Matched CPU-only metadata arms; preserve authentic mixed-TID observations."""
import os
import sys
from . import contract as C


def main():
    if any(name in sys.modules for name in ("torch", "numpy", "cv2")):
        raise RuntimeError("metadata bootstrap must precede numerical imports")
    os.environ.update(C.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W, snapshot as S
    from rl_agent.splitfusion_live_gt_placement_trial_v1.engine import Engine as Base
    from rl_agent.splitfusion_live_gt_placement_v1.placement import scheduling, process_threads
    from .candidate import make_engine_class
    candidate = make_engine_class(Base)
    expected = C.sources()
    old_engine, old_context, old_unpack = W.Engine, W.verify_child_context, S.unpack_snapshot
    owned = []
    numerical_context = None

    def construct(*, metadata_mode, **kwargs):
        W.require(metadata_mode in C.METADATA_MODES, "unknown metadata mode")
        W.require(kwargs.get("placement_mode") == "p4", "retained P4 scope required")
        engine = (Base if metadata_mode == "control" else candidate)(**kwargs)
        engine.metadata_mode = metadata_mode
        owned.append(engine)
        return engine

    def context(original_expected):
        nonlocal numerical_context
        import torch
        import cv2
        W.require(len(owned) == 1, "one dedicated GT owner required")
        engine = owned[0]
        placement = engine.placement
        if not placement.closed:
            placement.end_request("FAILED" if engine.failed else "OK")
        W.require(C.sources() == expected, "metadata sources changed")
        W.require(original_expected == placement.original_scheduling, "parent binding differs")
        if not placement.closed:
            rows = placement.verify()
        else:
            rows = process_threads(placement.pid)
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
        threads = {tid: {key: row[key] for key in ("affinity", "scheduler", "nice")}
                   for tid, row in rows.items()}
        result = dict(numeric, pid=os.getpid(), scheduling=actual, threads=threads,
            runtime_gc=engine.gc_policy.snapshot(include_events=False),
            runtime_placement=placement.snapshot(), original_parent_scheduling=original_expected,
            thread_context_validator="explicit_trial_main_and_helpers_original",
            runtime_metadata=dict(schema=C.METADATA_SCHEMA, mode=engine.metadata_mode,
                pid=os.getpid(), validators_removed=False, numerical_changes=False,
                counters=dict(getattr(engine, "metadata_counters", {}))))
        return C.validate_context(dict(result, bootstrap_sources=expected), expected)

    def unpack(payload):
        W.require(len(owned) == 1, "snapshot needs one GT owner")
        placement = owned[0].placement
        placement.begin_request()
        try:
            result = old_unpack(payload)
            placement.identify_request(result[0]["identity"])
            return result
        except BaseException:
            placement.end_request("SNAPSHOT_FAILED")
            raise

    W.Engine, W.verify_child_context, S.unpack_snapshot = construct, context, unpack
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

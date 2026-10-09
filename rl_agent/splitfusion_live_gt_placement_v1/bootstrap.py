"""Fresh child only: retain sealed request loop, GC control and numerical work."""
import os
import sys

from . import contract as C


def main():
    if any(name in sys.modules for name in ("torch", "numpy", "cv2")):
        raise RuntimeError("placement bootstrap must precede numerical imports")
    os.environ.update(C.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W, snapshot as S
    from .engine import Engine
    expected = C.sources()
    old_engine, old_context, old_unpack = W.Engine, W.verify_child_context, S.unpack_snapshot
    owned = []

    class Observed(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            owned.append(self)

    def context(original_expected):
        W.require(len(owned) == 1, "one dedicated GT placement engine required")
        engine = owned[0]
        placement = engine.placement
        if not placement.closed:
            placement.end_request("FAILED" if engine.failed else "OK")
        W.require(C.sources() == expected, "registered placement sources changed")
        W.require(original_expected == placement.original_scheduling,
                  "original parent scheduling binding differs")
        if not placement.closed:
            placement.verify()
        effective = placement.original_scheduling if placement.closed else placement.active_scheduling
        result = old_context(effective)
        result["runtime_gc"] = engine.gc_policy.snapshot(include_events=False)
        result["runtime_placement"] = placement.snapshot()
        result["original_parent_scheduling"] = original_expected
        return C.validate_context(dict(result, bootstrap_sources=expected), expected)

    def unpack(payload):
        W.require(len(owned) == 1, "snapshot needs one GT placement owner")
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

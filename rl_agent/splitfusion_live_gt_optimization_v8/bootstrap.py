"""Process-local candidate startup; never controls parent or edge GC."""
import os
import sys
from . import contract as C


def main(mode="freeze", contract=C):
    if any(name in sys.modules for name in ("torch", "numpy", "cv2")):
        raise RuntimeError("GT bootstrap must precede numerical imports")
    os.environ.update(contract.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
    from .engine import Engine, ControlEngine
    base = Engine if mode == "freeze" else ControlEngine
    expected = contract.sources()
    old_engine, old_context = W.Engine, W.verify_child_context
    owned = []

    class Observed(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            owned.append(self)

    def context(wanted):
        if contract.sources() != expected:
            raise ValueError("registered GC candidate sources changed")
        result = old_context(wanted)
        if owned:
            if len(owned) != 1:
                raise ValueError("one dedicated GT engine required")
            result["runtime_gc"] = owned[0].gc_policy.snapshot(include_events=False)
        return contract.validate_context(dict(result, bootstrap_sources=expected), expected)

    W.Engine, W.verify_child_context = Observed, context
    try:
        return W.main()
    finally:
        W.Engine, W.verify_child_context = old_engine, old_context
        for engine in owned:
            if engine.gc_policy is not None and not engine.gc_policy.closed:
                engine.gc_policy.close()


if __name__ == "__main__":
    raise SystemExit(main())

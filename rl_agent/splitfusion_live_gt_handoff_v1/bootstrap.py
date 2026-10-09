"""Caps before imports; no live GPU or scene ownership in this CPU process."""
import os
import sys
from . import contract as C


def main():
    if any(n in sys.modules for n in ("torch", "numpy", "cv2")):
        raise RuntimeError("GT bootstrap must precede numerical imports")
    os.environ.update(C.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
    from .engine import Engine
    expected = C.sources()
    old_engine, old_context = W.Engine, W.verify_child_context

    def context(wanted):
        if C.sources() != expected:
            raise ValueError("handoff sources changed")
        return C.validate_context(dict(old_context(wanted), bootstrap_sources=expected), expected)

    W.Engine, W.verify_child_context = Engine, context
    try:
        return W.main()
    finally:
        W.Engine, W.verify_child_context = old_engine, old_context


if __name__ == "__main__":
    raise SystemExit(main())

"""Dedicated CPU bootstrap; retained caps precede every numerical import."""
import os
import sys
from . import contract as C


def main():
    if any(name in sys.modules for name in ("torch", "numpy", "cv2")):
        raise RuntimeError("dedicated GT bootstrap must precede numerical imports")
    os.environ.update(C.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
    from .engine import Engine
    expected_sources = C.sources()
    original_context, original_engine = W.verify_child_context, W.Engine

    def checked_context(expected):
        if C.sources() != expected_sources:
            raise ValueError("dedicated GT candidate sources changed")
        return C.validate_context(dict(original_context(expected),
            bootstrap_sources=expected_sources), expected_sources)

    W.verify_child_context, W.Engine = checked_context, Engine
    try:
        return W.main()
    finally:
        W.verify_child_context, W.Engine = original_context, original_engine


if __name__ == "__main__":
    raise SystemExit(main())

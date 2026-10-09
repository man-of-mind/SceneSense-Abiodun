"""Reassert only dedicated-GT child caps before its first numerical import.

The retained UE launcher intentionally restores environments of ordinary
subprocesses. Do not disable or monkeypatch that hook. This module executes
only in the fresh dedicated worker, and then uses the unchanged V4 server.
"""
import os
import sys

from . import contract as C


def main():
    if any(name in sys.modules for name in ("torch", "numpy", "cv2")):
        raise RuntimeError("dedicated GT bootstrap must precede numerical imports")
    os.environ.update(C.ENVIRONMENT)
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
    expected_sources = C.sources()
    original = W.verify_child_context

    def checked_context(expected):
        if C.sources() != expected_sources:
            raise ValueError("dedicated GT bootstrap sources changed")
        context = dict(original(expected), bootstrap_sources=expected_sources)
        return C.validate_context(context, expected_sources)

    W.verify_child_context = checked_context
    try:
        return W.main()
    finally:
        W.verify_child_context = original


if __name__ == "__main__":
    raise SystemExit(main())

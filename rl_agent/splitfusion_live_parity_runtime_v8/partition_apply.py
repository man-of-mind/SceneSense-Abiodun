"""Exact retained affinity application, with only the new entry matcher."""
from pathlib import Path
import sys
from types import FunctionType
from .contract import EDGE


def apply(output, retained_code):
    from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as original
    fn = original.apply
    rebound = FunctionType(fn.__code__, dict(fn.__globals__, EDGE_MODULE=EDGE),
                           fn.__name__, fn.__defaults__, fn.__closure__)
    return rebound(output, retained_code)


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

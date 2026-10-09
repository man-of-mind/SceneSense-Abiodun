"""Private exact retained affinity application with the v2 command matcher."""
from pathlib import Path
import sys
from types import FunctionType
from .contract import EDGE


def apply(output, retained_code):
    from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as original
    function = original.apply
    rebound = FunctionType(function.__code__, dict(function.__globals__, EDGE_MODULE=EDGE),
                           function.__name__, function.__defaults__, function.__closure__)
    return rebound(output, retained_code)


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

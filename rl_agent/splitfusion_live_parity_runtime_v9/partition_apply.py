"""Unchanged retained partition application, with the new edge entry name."""
from pathlib import Path
import sys
from rl_agent.splitfusion_live_parity_runtime_v8 import partition_apply as V8
from .bindings import function
from .contract import EDGE

apply = function(V8.apply, dict(V8.apply.__globals__, EDGE=EDGE))

if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

"""Only the edge command matcher changes; all retained ownership checks remain."""
from pathlib import Path
import sys
from .contract import EDGE


def apply(output, retained_code):
    from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as original
    before = original.EDGE_MODULE
    original.EDGE_MODULE = EDGE
    try:
        return original.apply(output, retained_code)
    finally:
        original.EDGE_MODULE = before


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

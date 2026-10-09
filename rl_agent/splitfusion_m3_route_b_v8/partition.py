"""V8 partition identity (edge command-line matcher = the v8 edge trampoline); retained partition mechanics."""
from pathlib import Path
import sys

from .contract import EDGE


def apply(output, retained_code):
    from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as retained
    original = retained.EDGE_MODULE
    retained.EDGE_MODULE = EDGE
    try:
        return retained.apply(output, retained_code)
    finally:
        retained.EDGE_MODULE = original


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

# v13 provenance: copy of rl_agent/splitfusion_m3_route_b_v12/partition.py with only package/identity names mapped (v12 -> v13);
# it reads splitfusion_static_modes_v13.contract. Docstrings below are the V12 text and name V12 where they describe history.
"""V12 partition identity (edge command-line matcher = the v12 edge trampoline); retained partition mechanics."""
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

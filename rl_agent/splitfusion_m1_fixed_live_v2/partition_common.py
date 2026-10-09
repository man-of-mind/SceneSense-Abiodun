"""Bind the retained disjoint CPU partition to one fixed edge identity."""
from pathlib import Path


def apply(C, output, retained_code):
    from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as retained
    original = retained.EDGE_MODULE
    retained.EDGE_MODULE = C.EDGE
    try:
        return retained.apply(Path(output), Path(retained_code))
    finally:
        retained.EDGE_MODULE = original


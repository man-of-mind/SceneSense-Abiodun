"""Sampler-visible static-modes v13 edge (``--m3-verify-startup``: registration validation only, then exit)."""
import sys

from rl_agent.splitfusion_static_modes_v13.entries import edge, verify_edge_startup

if __name__ == "__main__":
    if sys.argv[1:2] == ["--m3-verify-startup"]:
        raise SystemExit(verify_edge_startup(sys.argv[2:]))
    raise SystemExit(edge())

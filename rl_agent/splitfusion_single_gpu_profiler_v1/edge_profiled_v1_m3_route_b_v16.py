"""Sampler-visible M3 v16 edge (``--m3-verify-startup``: registration validation only, then exit)."""
import sys

from rl_agent.splitfusion_m3_route_b_v16.entries import edge, verify_edge_startup

if __name__ == "__main__":
    if sys.argv[1:2] == ["--m3-verify-startup"]:
        raise SystemExit(verify_edge_startup(sys.argv[2:]))
    raise SystemExit(edge())

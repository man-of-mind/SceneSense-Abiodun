"""Sampler-visible M3 v20 child (``--m3-verify-startup --output <root>``: startup validation only, then exit)."""
import sys

from rl_agent.splitfusion_m3_route_b_v20.entries import child, verify_child_startup

if __name__ == "__main__":
    if sys.argv[1:2] == ["--m3-verify-startup"]:
        raise SystemExit(verify_child_startup(sys.argv[2:]))
    raise SystemExit(child())

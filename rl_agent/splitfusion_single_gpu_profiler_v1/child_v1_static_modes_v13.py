"""Sampler-visible static-modes v13 child (``--m3-verify-startup --output <root>``: startup validation only, then exit)."""
import sys

from rl_agent.splitfusion_static_modes_v13.entries import child, verify_child_startup

if __name__ == "__main__":
    if sys.argv[1:2] == ["--m3-verify-startup"]:
        raise SystemExit(verify_child_startup(sys.argv[2:]))
    raise SystemExit(child())

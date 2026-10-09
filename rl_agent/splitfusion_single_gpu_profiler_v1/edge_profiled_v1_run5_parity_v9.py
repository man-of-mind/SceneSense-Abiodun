"""Registered stdlib-first edge shim."""
from rl_agent.splitfusion_live_parity_runtime_v9.startup import edge_entry

if __name__ == "__main__":
    raise SystemExit(edge_entry())

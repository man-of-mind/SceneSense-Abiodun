"""Registered child shim; all lifecycle behavior belongs to the v9 adapter."""
from rl_agent.splitfusion_live_parity_runtime_v9.entries import child

if __name__ == "__main__":
    raise SystemExit(child())

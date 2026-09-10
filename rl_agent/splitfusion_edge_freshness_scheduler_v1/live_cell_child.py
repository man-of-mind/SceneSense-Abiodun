"""Run the timing live child with scheduler-aware UE feedback wrappers."""

from __future__ import annotations

from rl_agent.splitfusion_timing_diagnostic_v1 import live_cell_child as child

from .live_capture import install_live_wrappers


def main(argv: list[str] | None = None) -> int:
    child.install_live_wrappers = install_live_wrappers
    return child.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

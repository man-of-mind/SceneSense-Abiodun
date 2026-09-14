#!/usr/bin/env python3
"""Focused map-process test for LOCAL integration."""

from __future__ import annotations

from rl_agent.splitfusion_map_freshness_analysis_v1.integrate_local import map_metrics


def main() -> int:
    # The production helper requires the registered 300-frame inventory; make
    # a simple 10-Hz series with 50-ms installs.
    rows = [
        {"ack_status": "ACK_INSTALLED", "capture_raw_ns": str(i * 100_000_000), "edge_install_raw_ns": str(i * 100_000_000 + 50_000_000)}
        for i in range(300)
    ]
    result = map_metrics(rows, 0.0)
    assert result["fresh_map_time_ms_le_150_fraction"] > 0.98
    shifted = map_metrics(rows, 50.0)
    assert shifted["fresh_map_time_ms_le_150_fraction"] < result["fresh_map_time_ms_le_150_fraction"]
    assert shifted["fresh_map_time_ms_le_250_fraction"] > 0.98
    print("LOCAL integration tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Matched live CARLA/OAI comparison of v2 and final v3 edge service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common

from . import live_payoff_runner as runner
from .pipeline import CandidatePolicy


EXECUTE_TOKEN = "SPLITFUSION_EDGE_V3_FINAL_LIVE_COMPARISON_V1"
SUCCESS_TERMINAL = "SPLITFUSION_EDGE_V3_FINAL_LIVE_COMPARISON_COMPLETE"
DEFAULT_OUTPUT = (
    "experiments/splitfusion_edge_optimization_v3/"
    "20260910_live_v2_vs_v3_actions30_15_50_71"
)
VARIANTS = (
    {
        "id": "V2_PREDICTED_INSTALL_HORIZON",
        "policy": CandidatePolicy.PREDICTED_INSTALL_HORIZON,
        "edge_module": (
            "rl_agent.splitfusion_edge_freshness_scheduler_v1."
            "edge_service_v2_predicted"
        ),
        "tail": "SYNCHRONIZATION_LIGHT_V2",
    },
    {
        "id": "V3_OVERLAPPED_FINAL",
        "policy": CandidatePolicy.PREDICTED_INSTALL_HORIZON,
        "edge_module": (
            "rl_agent.splitfusion_edge_freshness_scheduler_v1."
            "edge_service_v3_predicted"
        ),
        "tail": "OVERLAPPED_OUTPUT_PRESERVING_V3",
    },
)
SOURCES = (
    *runner.SOURCES,
    "rl_agent/splitfusion_edge_optimization_v1/codec_v3.py",
    "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v3.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_tail_v3.py",
    "rl_agent/splitfusion_edge_optimization_v1/detached_edge_preload_v3.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/edge_service_v3_predicted.py",
    "rl_agent/splitfusion_edge_freshness_scheduler_v1/live_final_optimization_runner.py",
)


def _report(root: Path, rows: Sequence[Mapping[str, Any]], runtime_s: float) -> None:
    lines = [
        "# Final SplitFusion edge optimization: live v2 versus v3",
        "",
        "Each cell used 300 transmitted frames from live CARLA Route B and a fresh "
        "`FAVORABLE_STABLE` OAI/RFsim lifecycle. Both variants used the same "
        "causal predicted-install scheduler; only the edge implementation changed.",
        "",
        "| action | variant | installed | edge service med (ms) | "
        "install AoI med (ms) | queue med (ms) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def show(name: str) -> str:
            value = row.get(name)
            return "—" if value in (None, "") else f"{float(value):.2f}"

        lines.append(
            f"| {row['action_id']} | `{row['variant']}` | "
            f"{row['ack_installed_frames']} | "
            f"{show('edge_service_to_result_ms_median')} | "
            f"{show('install_aoi_ms_median')} | "
            f"{show('edge_queue_wait_ms_median')} |"
        )
    lines.extend(
        [
            "",
            f"Total wall time: {runtime_s / 60.0:.1f} minutes.",
            "",
            "The comparison does not claim 100 ms service readiness or "
            "run-to-run variance characterization.",
            "",
        ]
    )
    common.atomic_create_text(root / "REPORT.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    runner.EXECUTE_TOKEN = EXECUTE_TOKEN
    runner.SUCCESS_TERMINAL = SUCCESS_TERMINAL
    runner.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    runner.VARIANTS = VARIANTS
    runner.SOURCES = SOURCES
    runner._report = _report
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

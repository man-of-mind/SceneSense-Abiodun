#!/usr/bin/env python3
"""Focused CPU checks for the dual-clock timing analysis."""

from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageFrame,
)
from rl_agent.splitfusion_map_freshness_analysis_v1.analyze_timing_boundaries import (
    enforce_imputed_arrival_causality,
    wall_seconds_to_ns,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def frame(sequence: int, arrival_ns: int | None) -> TwoStageFrame:
    return TwoStageFrame(
        frame_id=100 + sequence,
        sequence_id=sequence,
        capture_ns=1_000_000_000,
        arrival_ns=arrival_ns,
        compute_ns=10_000_000,
        publication_ns=1_000_000,
        post_publication_install_ns=2_000_000,
        feature_bytes=100,
    )


def main() -> None:
    check(wall_seconds_to_ns("1.000000001") == 1_000_000_001, "decimal wall conversion")
    frames = [
        frame(0, 1_020_000_000),
        frame(1, 1_080_000_000),
        frame(2, None),
    ]
    rows = [
        {
            "ue_prepare_finished_ns": "50000000",
            "send_finished_ns": "90000000",
            "edge_receipt_wall_s": "",
        },
        {
            "ue_prepare_finished_ns": "50000000",
            "send_finished_ns": "90000000",
            "edge_receipt_wall_s": "1.08",
        },
        {
            "ue_prepare_finished_ns": "50000000",
            "send_finished_ns": "90000000",
            "edge_receipt_wall_s": "",
        },
    ]
    corrected, count, delta = enforce_imputed_arrival_causality(
        frames,
        rows,
        bridge_ns=1_000_000_000,
        label="test",
    )
    check(corrected[0].arrival_ns == 1_050_000_000, "imputed arrival floor")
    check(corrected[1].arrival_ns == 1_080_000_000, "observed arrival unchanged")
    check(corrected[2].arrival_ns is None, "missing transport remains missing")
    check(count == 1 and delta == 30_000_000, "floor accounting")
    # The floor is transmission start, not send-loop completion.
    check(corrected[0].arrival_ns < 1_090_000_000, "send completion was not used")
    print("test_timing_boundaries: PASS")


if __name__ == "__main__":
    main()

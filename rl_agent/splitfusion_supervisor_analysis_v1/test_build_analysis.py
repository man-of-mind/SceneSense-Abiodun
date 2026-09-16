#!/usr/bin/env python3
"""Small deterministic unit checks for analysis semantics."""

from __future__ import annotations

import math
from types import SimpleNamespace

from rl_agent.splitfusion_supervisor_analysis_v1.build_analysis import (
    PROFILE_ORDER,
    SCATTER_STAGES,
    STAGES,
    action_balanced_latency,
    edge_model_ready_components,
    equal_percentile_map,
    pareto_ids,
    percentile,
    quality_score,
    retained_pre_frozen_ms,
    sensor_optimization_stage_rows,
    shift_arrivals_for_sensor_optimization,
    useful_outcomes,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageFrame,
)


def test_quality_is_conservative_geometric_mean() -> None:
    overlap, centroid_rms_m, centroid_score, localization, combined = quality_score(
        {
            "val_segmentation_miou": 0.81,
            "val_vehicle_iou": 0.64,
            "val_person_box_mask_iou": 0.25,
            "val_vehicle_xy_mae_m": 0.6,
            "val_canonical_person_xy_mae_m": 0.8,
        }
    )
    assert abs(overlap - 0.4) < 1e-12
    assert abs(centroid_rms_m - math.sqrt(0.5)) < 1e-12
    assert abs(centroid_score - math.exp(-math.sqrt(0.5))) < 1e-12
    assert abs(localization - math.sqrt(overlap * centroid_score)) < 1e-12
    assert abs(combined - math.sqrt(0.81 * localization)) < 1e-12


def test_percentile_is_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 0.99) == 99.0


def test_model_ready_boundary_includes_required_reconstruction() -> None:
    row = {
        "edge_zstd_decompression_ms": 1.0,
        "edge_unpack_dequantize_ms": 4.0,
        "edge_ae_decode_ms": 2.0,
        "tail_camera_pose_reconstruct_ms": 0.5,
        "decode_tail_cuda_ms": 21.0,
        "decode_tail_inference_block_ms": 19.0,
        "edge_frozen_tail_ms": 37.5,
        "edge_total_edge_processing_ms": 50.0,
        "edge_output_serialization_ms": 5.0,
    }
    assert edge_model_ready_components(row) == (13.0, 21.0, 50.0)
    incoherent = dict(row, edge_total_edge_processing_ms=25.0)
    assert edge_model_ready_components(incoherent) is None


def test_scatter_endpoint_stops_at_model_tail_not_map_install() -> None:
    assert SCATTER_STAGES == (
        "ue_action",
        "network",
        "edge_model_ready",
        "sensor_model_ready",
    )


def test_retained_pre_frozen_preserves_action_specific_work() -> None:
    row = {
        "edge_timing_ns": (
            "{'zstd_decompression': 1000000, 'unpack_dequantize': 4000000, "
            "'ae_decode': 2000000, 'frozen_tail': 21000000, "
            "'output_serialization': 3000000, 'total_edge_processing': 30000000}"
        )
    }
    assert retained_pre_frozen_ms(row) == 6.0
    assert retained_pre_frozen_ms({"edge_timing_ns": ""}) is None


def test_pareto_minimizes_latency_and_maximizes_quality() -> None:
    rows = [
        {"action_id": 1, "x": 10.0, "combined_quality": 0.5},
        {"action_id": 2, "x": 20.0, "combined_quality": 0.4},
        {"action_id": 3, "x": 30.0, "combined_quality": 0.8},
    ]
    assert pareto_ids(rows, "x") == {1, 3}


def test_equal_percentile_map_preserves_distribution_rank() -> None:
    baseline = [10.0, 20.0, 30.0, 40.0]
    optimized = [1.0, 2.0, 3.0, 4.0]
    assert equal_percentile_map(10.0, baseline, optimized) == 1.0
    assert equal_percentile_map(25.0, baseline, optimized) == 2.0
    assert equal_percentile_map(40.0, baseline, optimized) == 4.0


def outcome(sequence: int, capture: int, install: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        frame=SimpleNamespace(sequence_id=sequence, capture_ns=capture),
        install_ns=install,
    )


def test_useful_installations_drop_older_late_map() -> None:
    result = SimpleNamespace(
        observation_end_ns=1_000,
        outcomes=(
            outcome(0, 100, 400),
            outcome(1, 300, 500),
            outcome(2, 200, 600),
            outcome(3, 700, None),
        ),
    )
    assert [item.frame.sequence_id for item in useful_outcomes(result)] == [0, 1]


def test_latency_percentiles_use_common_action_support() -> None:
    rows = []
    for profile_index, profile in enumerate(PROFILE_ORDER):
        for action_id in (0, 1):
            row = {"network_profile": profile, "action_id": action_id}
            for stage in STAGES:
                row[f"{stage}_count"] = 200
                for suffix in (50, 95, 99):
                    row[f"{stage}_p{suffix}_ms"] = float(
                        10 * profile_index + action_id + suffix
                    )
            if profile == "ADVERSE_STABLE" and action_id == 1:
                row["network_count"] = 4
            rows.append(row)
    summaries = action_balanced_latency(rows)
    assert all(row["support_actions"] == 1 for row in summaries)
    assert all(row["actions_with_p99"] == 1 for row in summaries)
    network = [row for row in summaries if row["stage"] == "network"]
    assert [row["action_balanced_p99_ms"] for row in network] == [99, 109, 119, 129]


def test_imputed_scheduler_arrival_cannot_precede_shifted_send() -> None:
    frame = TwoStageFrame(
        frame_id=1,
        sequence_id=0,
        capture_ns=1_000_000_000,
        arrival_ns=1_020_000_000,
        compute_ns=1,
        publication_ns=1,
        post_publication_install_ns=0,
        feature_bytes=1,
    )
    rebuilt, _, production, concat, deltas, floor_count = (
        shift_arrivals_for_sensor_optimization(
            [frame],
            [
                {
                    "pre_front_compute_ms": "10",
                    "scene_snapshot_ms": "0",
                    "send_finished_ns": "1030000000",
                }
            ],
            {
                "baseline_pre_action_ms": [10.0],
                "optimized_pre_action_ms": [5.0],
                "optimized_production_ms": [4.0],
                "optimized_concatenation_ms": [1.0],
            },
            0,
        )
    )
    assert rebuilt[0].arrival_ns == 1_025_000_000
    assert production == [4.0]
    assert concat == [1.0]
    assert deltas == [-5.0]
    assert floor_count == 1


def test_sensor_presentation_stages_are_sequential_and_auditable() -> None:
    rows = sensor_optimization_stage_rows()
    assert [row["stage"] for row in rows] == [
        f"P{index:02d}" for index in range(1, len(rows) + 1)
    ]
    assert rows[0]["source_stage"] == "P07"
    assert rows[-1]["source_stage"] == "P25"


if __name__ == "__main__":
    test_quality_is_conservative_geometric_mean()
    test_percentile_is_nearest_rank()
    test_model_ready_boundary_includes_required_reconstruction()
    test_scatter_endpoint_stops_at_model_tail_not_map_install()
    test_retained_pre_frozen_preserves_action_specific_work()
    test_pareto_minimizes_latency_and_maximizes_quality()
    test_equal_percentile_map_preserves_distribution_rank()
    test_useful_installations_drop_older_late_map()
    test_latency_percentiles_use_common_action_support()
    test_imputed_scheduler_arrival_cannot_precede_shifted_send()
    test_sensor_presentation_stages_are_sequential_and_auditable()
    print("SUPERVISOR_ANALYSIS_UNIT_TEST_PASS")

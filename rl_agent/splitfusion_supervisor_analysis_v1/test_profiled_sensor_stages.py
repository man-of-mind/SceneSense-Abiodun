#!/usr/bin/env python3
"""Focused equivalence tests for additive sensor-stage profiling."""

from __future__ import annotations

import numpy as np
import torch
import cv2

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion import radar_fusion
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _prepare_live_input
from rl_agent.splitfusion_supervisor_analysis_v1.profiled_sensor_stages import (
    build_radar_sample_profiled,
    prepare_live_input_profiled,
    profile_current_sweep_p40,
)
from rl_agent.splitfusion_hybrid_sac_v1.scene_descriptors import (
    camera_spatial_information,
)
from rl_agent.splitfusion_supervisor_analysis_v1.run_sensor_preparation_cell_v1 import (
    clock_bridge,
    complete_publication_join,
    full_path_metrics,
    sensor_stage_distributions,
)
from rl_agent import ue_288_campaign_supervisor as supervisor
from rl_agent.splitfusion_direct_edge_map_live_validation_v1 import (
    DIRECT_STAGE_INTERVALS,
)


def radar_kwargs(tracker: radar_fusion.StationaryTrackAccumulator) -> dict[str, object]:
    rng = np.random.default_rng(310)
    detections = np.column_stack(
        (
            rng.uniform(-0.2, 0.2, 128),
            rng.uniform(-0.8, 0.8, 128),
            rng.uniform(1.0, 100.0, 128),
            rng.uniform(-10.0, 10.0, 128),
        )
    ).astype(np.float32)
    intrinsics = np.array([[400.0, 0.0, 384.0], [0.0, 400.0, 224.0], [0.0, 0.0, 1.0]])
    return {
        "detections": detections,
        "sensor_matrix": np.eye(4),
        "camera_inverse_matrix": np.eye(4),
        "camera_intrinsics": intrinsics,
        "width": 768,
        "height": 448,
        "frame_time_s": 10.0,
        "tracker": tracker,
        "max_range_m": 120.0,
        "max_abs_velocity_mps": 20.0,
        "parked_threshold_s": 5.0,
        "point_radius_px": 4,
        "rasterizer": "fast",
    }


def test_radar_equivalence() -> None:
    original = radar_fusion.build_radar_sample(
        **radar_kwargs(radar_fusion.StationaryTrackAccumulator())
    )
    profiled = build_radar_sample_profiled(
        **radar_kwargs(radar_fusion.StationaryTrackAccumulator())
    )
    np.testing.assert_array_equal(original[0], profiled[0])
    assert original[1].keys() == profiled[1].keys()
    for key in original[1]:
        np.testing.assert_array_equal(original[1][key], profiled[1][key])
    for key, value in original[2].items():
        assert profiled[2][key] == value
    assert all(
        np.isfinite(value) and value >= 0.0
        for key, value in profiled[2].items()
        if key.startswith("profile_")
    )


def test_seven_channel_equivalence_cpu() -> None:
    rng = np.random.default_rng(711)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    radar = rng.normal(size=(4, 432, 768)).astype(np.float32)
    expected = _prepare_live_input(frame, radar, torch.device("cpu"))
    actual, measurements = prepare_live_input_profiled(frame, radar, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == (1, 7, 448, 768)
    assert all(np.isfinite(value) and value >= 0.0 for value in measurements.values())
    assert measurements["profile_cuda_substage_synchronizations"] == 0.0
    assert (
        measurements["profile_seven_channel_production_enqueue_wall_ms"]
        <= measurements["profile_seven_channel_total_wall_ms"]
    )


def test_cached_normalization_constants_are_exact() -> None:
    rng = np.random.default_rng(712)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    radar = rng.normal(size=(4, 432, 768)).astype(np.float32)
    device = torch.device("cpu")
    constants = (
        torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
    )
    expected = _prepare_live_input(frame, radar, device)
    actual, measurements = prepare_live_input_profiled(
        frame, radar, device, normalization_constants=constants
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert measurements["profile_cuda_substage_synchronizations"] == 0.0


def test_scene_si_is_read_only_and_uses_prepared_resolution() -> None:
    rng = np.random.default_rng(713)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    radar = rng.normal(size=(4, 448, 768)).astype(np.float32)
    expected = _prepare_live_input(frame, radar, torch.device("cpu"))
    actual, measurements = prepare_live_input_profiled(
        frame,
        radar,
        torch.device("cpu"),
        skip_identity_radar_resize=True,
        scene_descriptor_enabled=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (768, 448), interpolation=cv2.INTER_LINEAR)
    expected_si = camera_spatial_information(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    )
    assert measurements["camera_si"] == expected_si
    assert measurements["scene_descriptor_camera_status"] == "VALID"
    assert measurements["profile_scene_luma_ms"] >= 0.0
    assert measurements["profile_scene_si_ms"] >= 0.0


def test_scene_descriptor_off_does_no_camera_work() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    radar = np.zeros((4, 448, 768), dtype=np.float32)
    _actual, measurements = prepare_live_input_profiled(
        frame,
        radar,
        torch.device("cpu"),
        skip_identity_radar_resize=True,
        scene_descriptor_enabled=False,
    )
    assert "camera_si" not in measurements
    assert "scene_descriptor_camera_status" not in measurements
    assert measurements["profile_scene_luma_ms"] == 0.0
    assert measurements["profile_scene_si_ms"] == 0.0


def test_p40_uses_only_current_100ms_sweep() -> None:
    metadata = {
        "raw_provenance": {
            "original_range_m": np.asarray(
                [10.0, 20.0, 1.0, 2.0], dtype=np.float32
            ),
            # The close 1/2-m returns belong to the previous sweep and must
            # not influence the current-scene descriptor.
            "sweep_offset": np.asarray([0, 0, 1, 1], dtype=np.uint8),
        }
    }
    result = profile_current_sweep_p40(metadata, enabled=True)
    assert result["scene_descriptor_radar_status"] == "VALID"
    assert result["scene_descriptor_current_sweep_returns"] == 2
    assert result["radar_p40"] == (0.75 + 0.5) / 2.0
    assert result["profile_scene_p40_ms"] >= 0.0


def test_p40_missing_current_sweep_is_explicit_not_zero() -> None:
    metadata = {
        "raw_provenance": {
            "original_range_m": np.asarray([5.0], dtype=np.float32),
            "sweep_offset": np.asarray([1], dtype=np.uint8),
        }
    }
    result = profile_current_sweep_p40(metadata, enabled=True)
    assert result["scene_descriptor_radar_status"] == "RadarUnavailableError"
    assert result["radar_p40"] == ""
    assert result["scene_descriptor_current_sweep_returns"] == 0


def test_complete_publication_join_requires_every_boundary() -> None:
    publication = {"stream_id": "ue-1", "frame_id": "17"}
    installed = {
        "stream_id": "ue-1",
        "frame_id": "17",
        "outcome": "RESULT_INSTALLED",
    }
    cursor = 1.0
    for _name, start, finish in DIRECT_STAGE_INTERVALS:
        publication.setdefault(start, str(cursor))
        cursor += 0.001
        installed.setdefault(finish, str(cursor))
        publication.setdefault(finish, str(cursor))
    complete = complete_publication_join([installed], [publication])
    assert complete["complete_frames"] == 1
    assert complete["complete_fraction"] == 1.0
    installed.pop("map_install_at")
    publication.pop("map_install_at", None)
    incomplete = complete_publication_join([installed], [publication])
    assert incomplete["complete_frames"] == 0
    assert incomplete["incomplete_examples"]


def test_direct_clock_availability_and_same_domain_metrics() -> None:
    sent = {
        "stream_id": "ue-1",
        "frame_id": "17",
        "capture_started_ns": "1000000000",
        "send_finished_ns": "1100000000",
        "capture_wall_s": "1000.0",
        # Deliberately nonsensical legacy fields prove they are not used.
        "feature_received_at": "9000.0",
        "edge_result_received_ns": "1",
    }
    publication = {
        "stream_id": "ue-1",
        "frame_id": "17",
        "reassembly_complete_wall_s": "1000.2",
        "compute_start_wall_s": "1000.21",
        "tail_complete_wall_s": "1000.24",
        "first_datagram_send_wall_s": "1000.245",
    }
    ingest = {
        "stream_id": "ue-1",
        "frame_id": "17",
        "outcome": "RESULT_INSTALLED",
        "map_ingest_at": "1000.25",
        "map_install_at": "1000.30",
    }
    unavailable = clock_bridge([sent])
    assert unavailable["availability"] == "UNAVAILABLE"
    assert unavailable["legacy_edge_result_fields_used"] is False
    result = full_path_metrics([sent], [ingest], [publication], unavailable)
    metrics = result["metrics"]
    assert result["cross_clock_subtraction_performed"] is False
    assert metrics["ue_action"]["availability"] == "AVAILABLE"
    assert metrics["ue_action"]["p50_ms"] == 100.0
    assert abs(metrics["edge_compute"]["p50_ms"] - 30.0) < 1e-6
    assert abs(metrics["map_service"]["p50_ms"] - 55.0) < 1e-6
    assert abs(metrics["capture_to_install_aoi"]["p50_ms"] - 300.0) < 1e-6
    assert metrics["feature_uplink"]["availability"] == "UNAVAILABLE"
    assert metrics["feature_uplink"]["p50_ms"] is None
    assert metrics["action_start_to_install"]["availability"] == "UNAVAILABLE"
    assert metrics["action_start_to_install"]["p50_ms"] is None

    anchored = dict(sent)
    anchored.update(
        {
            "ue_clock_anchor_wall_ns": "1000000000000",
            "ue_clock_anchor_perf_ns": "1000000000",
        }
    )
    available = clock_bridge([anchored])
    assert available["availability"] == "AVAILABLE"
    assert available["deviation_within_bound"] is True
    assert available["legacy_edge_result_fields_used"] is False


def test_required_sensor_evidence_still_fails_closed() -> None:
    try:
        sensor_stage_distributions([{"profile_sensor_compute_production_estimate_ms": "1.0"}])
    except supervisor.CampaignError as exc:
        assert "required sensor-stage evidence is absent" in str(exc)
    else:
        raise AssertionError("missing required sensor-stage evidence was accepted")


if __name__ == "__main__":
    test_radar_equivalence()
    test_seven_channel_equivalence_cpu()
    test_cached_normalization_constants_are_exact()
    test_complete_publication_join_requires_every_boundary()
    test_direct_clock_availability_and_same_domain_metrics()
    test_required_sensor_evidence_still_fails_closed()
    print("PROFILED_SENSOR_STAGES_TEST_PASS")

#!/usr/bin/env python3
"""Focused equivalence tests for additive sensor-stage profiling."""

from __future__ import annotations

import numpy as np
import torch

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion import radar_fusion
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _prepare_live_input
from rl_agent.splitfusion_supervisor_analysis_v1.profiled_sensor_stages import (
    build_radar_sample_profiled,
    prepare_live_input_profiled,
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


if __name__ == "__main__":
    test_radar_equivalence()
    test_seven_channel_equivalence_cpu()
    print("PROFILED_SENSOR_STAGES_TEST_PASS")

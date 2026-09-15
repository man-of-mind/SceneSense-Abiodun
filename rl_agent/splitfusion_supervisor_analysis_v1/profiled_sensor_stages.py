"""Bit-preserving profiling equivalents for the next short sensor run.

These functions are additive instrumentation. They do not replace either
hash-pinned production function in this commit and are not evidence until a
separately authorized live run exercises them.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion import radar_fusion


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1e6


def build_radar_sample_profiled(**kwargs: Any) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Equivalent to ``build_radar_sample`` with P1--P6 timings."""

    detections = kwargs["detections"]
    tracker = kwargs["tracker"]
    total_started = time.perf_counter_ns()

    started = time.perf_counter_ns()
    world_velocity = radar_fusion.radar_spherical_to_world(
        detections, kwargs["sensor_matrix"]
    )
    spherical_to_world_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    ages = tracker.update(world_velocity, kwargs["frame_time_s"])
    stationary_tracking_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    points_cam = (
        radar_fusion.world_to_camera_points(
            world_velocity[:, :3], kwargs["camera_inverse_matrix"]
        )
        if world_velocity.size
        else np.zeros((0, 3), dtype=np.float64)
    )
    world_to_camera_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    u, v, depth, valid = radar_fusion.project_camera_points(
        points_cam, kwargs["camera_intrinsics"]
    )
    projection_ms = _elapsed_ms(started)
    velocities = (
        world_velocity[:, 3].astype(np.float32)
        if world_velocity.size
        else np.zeros((0,), dtype=np.float32)
    )

    name = str(kwargs.get("rasterizer", "legacy") or "legacy").strip().lower()
    if name in ("legacy", "python"):
        rasterize = radar_fusion.rasterize_radar_channels
    elif name in ("fast", "vectorized"):
        rasterize = radar_fusion.rasterize_radar_channels_fast
    else:
        raise ValueError(f"unknown radar rasterizer {name!r}")
    started = time.perf_counter_ns()
    tensor = rasterize(
        width=kwargs["width"],
        height=kwargs["height"],
        u=u,
        v=v,
        depth_m=depth,
        velocity_mps=velocities,
        stationary_age_s=ages,
        valid_mask=valid,
        max_range_m=kwargs["max_range_m"],
        max_abs_velocity_mps=kwargs["max_abs_velocity_mps"],
        parked_threshold_s=kwargs["parked_threshold_s"],
        point_radius_px=kwargs["point_radius_px"],
    )
    rasterization_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    points = {
        "world_xyz": world_velocity[:, :3].astype(np.float32)
        if world_velocity.size
        else np.zeros((0, 3), dtype=np.float32),
        "camera_xyz": points_cam.astype(np.float32)
        if points_cam.size
        else np.zeros((0, 3), dtype=np.float32),
        "velocity_mps": velocities.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "camera_depth_m": depth.astype(np.float32),
        "stationary_age_s": ages.astype(np.float32),
        "valid_projection": valid.astype(np.uint8),
    }
    stationary = np.abs(velocities) <= float(tracker.stationary_velocity_mps)
    parked = ages >= float(kwargs["parked_threshold_s"])
    summary = {
        "radar_points": float(detections.shape[0]),
        "radar_stationary_points": float(np.sum(stationary)),
        "radar_parked_evidence_points": float(np.sum(parked)),
    }
    packaging_ms = _elapsed_ms(started)
    summary.update(
        {
            "profile_radar_spherical_to_world_ms": spherical_to_world_ms,
            "profile_radar_stationary_tracking_ms": stationary_tracking_ms,
            "profile_radar_world_to_camera_ms": world_to_camera_ms,
            "profile_radar_projection_ms": projection_ms,
            "profile_radar_rasterization_ms": rasterization_ms,
            "profile_radar_packaging_ms": packaging_ms,
            "profile_radar_total_ms": _elapsed_ms(total_started),
        }
    )
    return tensor, points, summary


def _timed_operation(
    name: str,
    operation: Any,
    device: torch.device,
    cpu_timings: dict[str, float],
    cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
) -> Any:
    """Run one stage without adding a per-stage CUDA synchronization."""

    if device.type != "cuda":
        started = time.perf_counter_ns()
        value = operation()
        cpu_timings[name] = _elapsed_ms(started)
        return value
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    value = operation()
    finish.record()
    cuda_events[name] = (start, finish)
    return value


def prepare_live_input_profiled(
    frame_bgr: np.ndarray,
    radar_tensor: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equivalent seven-channel preparation with CPU/CUDA stage timing.

    CUDA events introduce an intentional synchronization barrier for profiling;
    this function therefore belongs only in a short diagnostic, never in the
    production hot path.
    """

    total_started = time.perf_counter_ns()
    started = time.perf_counter_ns()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bgr_to_rgb_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    rgb = cv2.resize(rgb, (768, 448), interpolation=cv2.INTER_LINEAR)
    rgb_resize_ms = _elapsed_ms(started)

    started = time.perf_counter_ns()
    rgb_host = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    rgb_tensor_pack_ms = _elapsed_ms(started)
    cpu_timings: dict[str, float] = {}
    cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    rgb_tensor = _timed_operation(
        "profile_camera_h2d_ms",
        lambda: rgb_host.to(device=device, dtype=torch.float32).div_(255.0),
        device,
        cpu_timings,
        cuda_events,
    )
    constants_started = time.perf_counter_ns()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normalization_constants_wall_ms = _elapsed_ms(constants_started)
    rgb_normalized = _timed_operation(
        "profile_camera_normalize_ms",
        lambda: (rgb_tensor - mean) / std,
        device,
        cpu_timings,
        cuda_events,
    )

    started = time.perf_counter_ns()
    radar_host = torch.from_numpy(
        np.ascontiguousarray(
            np.stack(
                [
                    cv2.resize(
                        channel,
                        (768, 448),
                        interpolation=(cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR),
                    )
                    for index, channel in enumerate(radar_tensor)
                ],
                axis=0,
            )
        )
    ).unsqueeze(0)
    radar_resize_pack_ms = _elapsed_ms(started)
    radar_device = _timed_operation(
        "profile_radar_h2d_ms",
        lambda: radar_host.to(device=device, dtype=torch.float32),
        device,
        cpu_timings,
        cuda_events,
    )
    combined = _timed_operation(
        "profile_seven_channel_concatenate_ms",
        lambda: torch.cat((rgb_normalized, radar_device), dim=1),
        device,
        cpu_timings,
        cuda_events,
    )
    production_enqueue_wall_ms = _elapsed_ms(total_started)
    synchronization_started = time.perf_counter_ns()
    if cuda_events:
        # One barrier makes every event readable. It is intentionally confined
        # to the diagnostic path and is never inserted between GPU substages.
        next(reversed(cuda_events.values()))[1].synchronize()
        cpu_timings.update(
            {
                name: float(start.elapsed_time(finish))
                for name, (start, finish) in cuda_events.items()
            }
        )
    diagnostic_sync_wait_ms = _elapsed_ms(synchronization_started)
    return combined, {
        "profile_camera_bgr_to_rgb_ms": bgr_to_rgb_ms,
        "profile_camera_resize_ms": rgb_resize_ms,
        "profile_camera_tensor_pack_ms": rgb_tensor_pack_ms,
        "profile_camera_h2d_ms": cpu_timings["profile_camera_h2d_ms"],
        "profile_camera_normalization_constants_wall_ms": normalization_constants_wall_ms,
        "profile_camera_normalize_ms": cpu_timings["profile_camera_normalize_ms"],
        "profile_radar_resize_pack_ms": radar_resize_pack_ms,
        "profile_radar_h2d_ms": cpu_timings["profile_radar_h2d_ms"],
        "profile_seven_channel_concatenate_ms": cpu_timings[
            "profile_seven_channel_concatenate_ms"
        ],
        "profile_seven_channel_production_enqueue_wall_ms": production_enqueue_wall_ms,
        "profile_seven_channel_diagnostic_sync_wait_ms": diagnostic_sync_wait_ms,
        "profile_cuda_substage_synchronizations": float(1 if cuda_events else 0),
        "profile_seven_channel_total_wall_ms": _elapsed_ms(total_started),
    }

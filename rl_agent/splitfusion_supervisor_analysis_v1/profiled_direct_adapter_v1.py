#!/usr/bin/env python3
"""Additive P1--Pn sensor profiling over the qualified direct-map adapter.

The historical adapter and direct-map seams remain imported unchanged.  This
wrapper replaces only the collector class and the seven-channel preparation
call for a short diagnostic cell.  It never changes the sensor configuration,
radar-window membership, action, model input, wire path, or map path.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rl_agent.ue_route_b_split_cell_adapter_v1 as pinned  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as direct  # noqa: E402
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base  # noqa: E402
from rl_agent.splitfusion_supervisor_analysis_v1.profiled_sensor_stages import (  # noqa: E402
    build_radar_sample_profiled,
    prepare_live_input_profiled,
)


BASELINE_MODE = "INSTRUMENTED_PRODUCTION_EQUIVALENT"
OPTIMIZED_MODE = "OPTIMIZED_SENSOR_PREPARATION"
PROFILE_FIELDS = (
    "sensor_profile_mode",
    "profile_rgb_callback_ms",
    "profile_radar_callback_ms",
    "profile_semantic_callback_ms",
    "profile_worker_schedule_wait_ms",
    "profile_rgb_callback_to_worker_ms",
    "profile_visible_actor_count",
    "profile_ego_acceleration_mps2",
    "profile_ego_yaw_rate_deg_s",
    "profile_radar_spherical_to_world_ms",
    "profile_radar_stationary_tracking_ms",
    "profile_radar_world_to_camera_ms",
    "profile_radar_projection_ms",
    "profile_radar_rasterization_ms",
    "profile_radar_packaging_ms",
    "profile_radar_total_ms",
    "profile_camera_bgr_to_rgb_ms",
    "profile_camera_resize_ms",
    "profile_camera_tensor_pack_ms",
    "profile_camera_h2d_ms",
    "profile_camera_normalization_constants_wall_ms",
    "profile_camera_normalize_ms",
    "profile_radar_resize_pack_ms",
    "profile_radar_h2d_ms",
    "profile_seven_channel_concatenate_ms",
    "profile_seven_channel_production_enqueue_wall_ms",
    "profile_seven_channel_diagnostic_sync_wait_ms",
    "profile_cuda_substage_synchronizations",
    "profile_seven_channel_total_wall_ms",
    "profile_unattributed_pre_front_ms",
    "profile_sensor_compute_production_estimate_ms",
    "profile_sensor_compute_diagnostic_wall_ms",
    "profile_radar_tensor_exact",
    "profile_radar_evidence_exact",
    "profile_model_input_exact",
    "profile_equivalence_checked",
    "ue_clock_anchor_wall_ns",
    "ue_clock_anchor_perf_ns",
)


class _FrameRecorder:
    """Thread-aware timing/covariate handoff into the collector's CSV row."""

    def __init__(self) -> None:
        self.local = threading.local()
        self.lock = threading.Lock()
        self.values: dict[int, dict[str, Any]] = {}
        self.callbacks: dict[int, dict[str, float]] = {}

    def begin(self, frame_id: int) -> None:
        self.local.frame_id = int(frame_id)
        with self.lock:
            self.values[int(frame_id)] = {}

    def end(self) -> None:
        self.local.frame_id = None

    def current(self) -> int:
        value = getattr(self.local, "frame_id", None)
        if value is None:
            raise pinned.AdapterError("sensor profiler has no current frame identity")
        return int(value)

    def add(self, values: Mapping[str, Any]) -> None:
        frame_id = self.current()
        with self.lock:
            self.values.setdefault(frame_id, {}).update(dict(values))

    def callback(self, frame_id: int, field: str, elapsed_ms: float) -> None:
        with self.lock:
            self.callbacks.setdefault(int(frame_id), {})[str(field)] = float(elapsed_ms)
            cutoff = int(frame_id) - 256
            for old in [key for key in self.callbacks if key < cutoff]:
                self.callbacks.pop(old, None)

    def take(self, frame_id: int) -> dict[str, Any]:
        with self.lock:
            values = self.values.pop(int(frame_id), {})
            values.update(self.callbacks.pop(int(frame_id), {}))
            return values


_RECORDER: _FrameRecorder | None = None
_ORIGINAL_PREPARE = live_base._prepare_live_input


def _profiled_prepare(
    frame_bgr: np.ndarray, radar_tensor: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Profile the actual input and prove live exactness on preregistered frames."""

    if _RECORDER is None:
        raise pinned.AdapterError("sensor profile recorder was not installed")
    collector = getattr(_RECORDER.local, "collector", None)
    normalization_constants = (
        getattr(collector, "_normalization_constants", None)
        if collector is not None
        else None
    )
    output, timings = prepare_live_input_profiled(
        frame_bgr,
        radar_tensor,
        device,
        normalization_constants=normalization_constants,
    )
    frame_id = _RECORDER.current()
    checked = bool(
        collector is not None
        and int(getattr(collector, "_equivalence_seen", 0))
        < int(getattr(collector, "_equivalence_frames", 0))
    )
    exact = ""
    if checked:
        reference = _ORIGINAL_PREPARE(frame_bgr, radar_tensor, device)
        exact = bool(torch.equal(output, reference))
        if not exact:
            raise pinned.AdapterError(
                f"frame {frame_id}: profiled seven-channel input is not exact"
            )
    _RECORDER.add(
        {
            **timings,
            "profile_model_input_exact": exact,
        }
    )
    return output


class _ParkedProxy:
    """Delegate the historical module except for one profiled radar call."""

    def __init__(self, module: Any, collector: "ProfiledPassiveSplitCollector") -> None:
        self._module = module
        self._collector = collector

    def build_radar_sample(self, **kwargs: Any) -> Any:
        return self._collector._profile_radar_sample(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class ProfiledPassiveSplitCollector(pinned.PassiveSplitCollector):
    """The pinned collector plus additive, output-equivalent measurements."""

    def __init__(self, **kwargs: Any) -> None:
        campaign = kwargs["campaign"]
        contract = dict(campaign.get("_sensor_preparation_diagnostic") or {})
        self._mode = str(contract.get("mode") or "")
        pinned.require(
            self._mode in {BASELINE_MODE, OPTIMIZED_MODE},
            f"unsupported sensor preparation diagnostic mode: {contract.get('mode')!r}",
        )
        self._recorder = _FrameRecorder()
        global _RECORDER
        _RECORDER = self._recorder
        self._equivalence_frames = int(contract.get("equivalence_frames", 8))
        pinned.require(
            1 <= self._equivalence_frames <= 32,
            "live equivalence frame count must be in [1, 32]",
        )
        self._equivalence_seen = 0
        super().__init__(**kwargs)
        self._normalization_constants: tuple[torch.Tensor, torch.Tensor] | None = None
        if self._mode == OPTIMIZED_MODE:
            self._normalization_constants = (
                torch.tensor(
                    [0.485, 0.456, 0.406], device=self.live.device
                ).view(1, 3, 1, 1),
                torch.tensor(
                    [0.229, 0.224, 0.225], device=self.live.device
                ).view(1, 3, 1, 1),
            )
        original_parked = self.parked
        self._original_radar_builder = original_parked.build_radar_sample
        self._shadow_tracker = live_base.FastStationaryTrackAccumulator(
            stationary_velocity_mps=float(self.tracker.stationary_velocity_mps),
            parked_threshold_s=float(self.tracker.parked_threshold_s),
            association_grid_m=float(self.tracker.association_grid_m),
            max_stale_s=float(self.tracker.max_stale_s),
        )
        self.parked = _ParkedProxy(original_parked, self)
        if self.scene_source is not None:
            original_capture = self.scene_source.capture

            def capture_with_count(snapshot: Any) -> Any:
                frozen = original_capture(snapshot)
                actors = list(frozen.get_actors())
                self._recorder.add({"profile_visible_actor_count": len(actors)})
                return frozen

            self.scene_source.capture = capture_with_count  # type: ignore[method-assign]

    def _on_rgb(self, image: Any) -> None:
        started = time.perf_counter_ns()
        super()._on_rgb(image)
        self._recorder.callback(
            int(image.frame), "profile_rgb_callback_ms",
            (time.perf_counter_ns() - started) / 1e6,
        )

    def _on_radar(self, measurement: Any) -> None:
        started = time.perf_counter_ns()
        super()._on_radar(measurement)
        self._recorder.callback(
            int(measurement.frame), "profile_radar_callback_ms",
            (time.perf_counter_ns() - started) / 1e6,
        )

    def _on_semantic(self, image: Any) -> None:
        started = time.perf_counter_ns()
        super()._on_semantic(image)
        self._recorder.callback(
            int(image.frame), "profile_semantic_callback_ms",
            (time.perf_counter_ns() - started) / 1e6,
        )

    def _profile_radar_sample(self, **kwargs: Any) -> Any:
        tensor, points, summary = build_radar_sample_profiled(**kwargs)
        checked = self._equivalence_seen < self._equivalence_frames
        tensor_exact: Any = ""
        evidence_exact: Any = ""
        if checked:
            reference_kwargs = dict(kwargs)
            reference_kwargs["tracker"] = self._shadow_tracker
            reference_tensor, reference_points, reference_summary = (
                self._original_radar_builder(**reference_kwargs)
            )
            tensor_exact = bool(np.array_equal(tensor, reference_tensor))
            evidence_exact = bool(
                points.keys() == reference_points.keys()
                and all(
                    np.array_equal(points[name], reference_points[name])
                    for name in points
                )
                and all(
                    summary[name] == reference_summary[name]
                    for name in reference_summary
                )
            )
            if not tensor_exact or not evidence_exact:
                raise pinned.AdapterError(
                    f"frame {self._recorder.current()}: profiled radar output is not exact"
                )
        self._recorder.add(
            {
                **{
                    key: value
                    for key, value in summary.items()
                    if str(key).startswith("profile_")
                },
                "profile_radar_tensor_exact": tensor_exact,
                "profile_radar_evidence_exact": evidence_exact,
            }
        )
        return tensor, points, {
            key: value for key, value in summary.items()
            if not str(key).startswith("profile_")
        }

    @staticmethod
    def _magnitude(vector: Any) -> float:
        return float(
            np.sqrt(
                float(vector.x) ** 2
                + float(vector.y) ** 2
                + float(vector.z) ** 2
            )
        )

    def _process_token(self, token: Mapping[str, Any]) -> None:
        frame_id = int(token["frame_id"])
        # Same-process, adjacent clock reads provide the explicit bridge needed
        # for direct edge-to-map timing.  No edge-to-UE result path is assumed.
        clock_anchor_wall_ns = time.time_ns()
        clock_anchor_perf_ns = time.perf_counter_ns()
        process_started_ns = time.perf_counter_ns()
        self._recorder.begin(frame_id)
        self._recorder.local.collector = self
        self._recorder.add(
            {
                "sensor_profile_mode": self._mode,
                "ue_clock_anchor_wall_ns": clock_anchor_wall_ns,
                "ue_clock_anchor_perf_ns": clock_anchor_perf_ns,
                "profile_worker_schedule_wait_ms": max(
                    0.0,
                    process_started_ns / 1e9 - float(token["scheduled_perf"]),
                )
                * 1000.0,
            }
        )
        try:
            super()._process_token(token)
            acceleration = self.ego.get_acceleration()
            angular_velocity = self.ego.get_angular_velocity()
            self._recorder.add(
                {
                    "profile_ego_acceleration_mps2": self._magnitude(acceleration),
                    "profile_ego_yaw_rate_deg_s": float(angular_velocity.z),
                }
            )
        finally:
            values = self._recorder.take(frame_id)
            with self.rows_lock:
                row = next(
                    (
                        item for item in reversed(self.rows)
                        if int(item.get("frame_id", -1)) == frame_id
                    ),
                    None,
                )
                if row is not None:
                    sensor_wait_ms = float(row.get("sensor_wait_ms") or 0.0)
                    with self.sensor_condition:
                        image_record = self.images.get(frame_id)
                    if image_record is not None:
                        sensor_ready_ns = process_started_ns + int(
                            round(sensor_wait_ms * 1e6)
                        )
                        values["profile_rgb_callback_to_worker_ms"] = max(
                            0.0,
                            (sensor_ready_ns - int(round(float(image_record[2]) * 1e9)))
                            / 1e6,
                        )
                    pre_front = float(row.get("pre_front_compute_ms") or 0.0)
                    scene_snapshot = float(row.get("scene_snapshot_ms") or 0.0)
                    radar_total = float(values.get("profile_radar_total_ms") or 0.0)
                    radar_window = float(row.get("radar_window_ms") or 0.0)
                    rgb_convert = float(row.get("rgb_convert_ms") or 0.0)
                    values["profile_unattributed_pre_front_ms"] = max(
                        0.0,
                        pre_front
                        - radar_window
                        - radar_total
                        - rgb_convert
                        - scene_snapshot,
                    )
                    sensor_base = max(0.0, pre_front - scene_snapshot)
                    enqueue = float(
                        values.get(
                            "profile_seven_channel_production_enqueue_wall_ms"
                        )
                        or 0.0
                    )
                    diagnostic = float(
                        values.get("profile_seven_channel_total_wall_ms") or 0.0
                    )
                    values["profile_sensor_compute_production_estimate_ms"] = (
                        sensor_base + enqueue
                    )
                    values["profile_sensor_compute_diagnostic_wall_ms"] = (
                        sensor_base + diagnostic
                    )
                    checked = self._equivalence_seen < self._equivalence_frames
                    values["profile_equivalence_checked"] = bool(checked)
                    if checked and row.get("prepare_status") == "SENT":
                        pinned.require(
                            values.get("profile_radar_tensor_exact") is True
                            and values.get("profile_radar_evidence_exact") is True
                            and values.get("profile_model_input_exact") is True,
                            f"frame {frame_id}: incomplete exact-equivalence proof",
                        )
                        self._equivalence_seen += 1
                    row.update(values)
            self._recorder.local.collector = None
            self._recorder.end()


def install_sensor_profile_seams(campaign: Mapping[str, Any]) -> None:
    """Install the diagnostic seams after the qualified direct-map seams."""

    contract = dict(campaign.get("_sensor_preparation_diagnostic") or {})
    pinned.require(
        contract.get("mode") in {BASELINE_MODE, OPTIMIZED_MODE},
        "supported sensor-profile mode is required",
    )
    global _RECORDER
    _RECORDER = None
    live_base._prepare_live_input = _profiled_prepare
    pinned.PassiveSplitCollector = ProfiledPassiveSplitCollector
    pinned.PER_FRAME_FIELDS = tuple(
        list(pinned.PER_FRAME_FIELDS)
        + [name for name in PROFILE_FIELDS if name not in pinned.PER_FRAME_FIELDS]
    )


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    args = pinned.build_parser().parse_args(values)
    pinned.require(
        args.resolved_config is not None and args.attempt_dir is not None,
        "profiled direct run requires --resolved-config and --attempt-dir",
    )
    resolved = pinned.load_yaml(args.resolved_config.resolve(strict=True))
    campaign = resolved["campaign"]
    cell = resolved["cell"]
    direct._ENDPOINT["cell_id"] = str(cell["cell_id"])
    direct._ENDPOINT["attempt_dir"] = str(args.attempt_dir.resolve())
    direct._ENDPOINT["config_relpath"] = str(
        campaign["runtime"]["direct_edge_config_relpath"]
    )
    endpoint = direct.install_direct_seams(campaign)
    install_sensor_profile_seams(campaign)
    mode = str((campaign.get("_sensor_preparation_diagnostic") or {}).get("mode"))
    print(
        f"[SENSOR-PROFILE] mode={mode}; "
        "one end-of-pipeline CUDA timing synchronization; "
        f"direct map {endpoint['host']}:{endpoint['port']}",
        flush=True,
    )
    return pinned.main(values)


if __name__ == "__main__":
    raise SystemExit(main())

"""Radar-preparation bundle: connect the validated optimization to the Phase-6 path.

Three bit-exact components, all **reused** from
``rl_agent/splitfusion_sensor_optimization_v1/optimized_stages.py`` and the
validated composition in ``splitfusion_supervisor_analysis_v1/profiled_sensor_stages.py``
(``SENSOR_PREPARATION_OPTIMIZATION_VALIDATED``, 2026-09-14).  They change
together and are reported as one bundle, not as evidence for any one of them:

1. **CUDA scatter-reduce rasterizer** -- ``CudaRadarRasterizer`` handed to the
   validated ``build_radar_sample_profiled`` as ``rasterizer_override`` (the
   production ``build_radar_sample`` has no override parameter; that profiled
   builder is the validated composition and is proven output-identical to
   production by ``tests/test_wired_path_equivalence.py``).  Its ``profile_*``
   timing keys are removed so the returned summary has exactly the production
   keys.
2. **Single-sort stationary tracker** -- the collector's
   ``FastStationaryTrackAccumulator`` is replaced by
   ``SingleSortStationaryTrackAccumulator`` with the identical parameters,
   before it has accumulated any state.
3. **Identity radar-resize short circuit** -- the seven-channel packing skips
   the per-channel 768x448 -> 768x448 ``cv2.resize`` when
   ``radar_channels_already_sized`` holds.  The rest of the packing is the
   production body verbatim.  The validated ``prepare_live_input_profiled`` is
   *not* used live: its per-stage CUDA events are a synchronization barrier
   its own docstring confines to diagnostics.

What is not touched: radar windows, window membership, sweep count, timestamps,
the planner's window-digest binding, state commit / action open, the camera
source timestamp, SI/P40 (computed from the image and window metadata, not
from the rasterized tensor), the actor, reward, deadline, codec, model, map.
The Phase-6 ``radar_tensor_start/end`` stamps and the profiler's
``ue.radar_rasterize`` / ``ue.input_7ch`` spans still bracket the bundle.

Importing this module initializes nothing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

BUNDLE_ENV = "SCENESENSE_RADAR_BUNDLE"
PRODUCTION, OPTIMIZED = "production", "optimized"
COMPONENTS = ("cuda_scatter_reduce_rasterizer", "single_sort_stationary_tracker",
              "identity_radar_resize_short_circuit")
PROFILE_KEY_PREFIX = "profile_"
SNAPSHOT_NAME = "RADAR_BUNDLE.json"
MODEL_WIDTH, MODEL_HEIGHT = 768, 448


class RadarBundleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RadarBundleError(message)


# ---------------------------------------------------------------------------
# component 3: packing with the identity-resize short circuit
# ---------------------------------------------------------------------------


def prepare_live_input_identity_resize(frame_bgr: np.ndarray, radar_tensor: np.ndarray,
                                       device: Any) -> Any:
    """``live_pilot_runtime._prepare_live_input`` with only the identity resize skipped."""
    import cv2
    import torch

    from rl_agent.splitfusion_sensor_optimization_v1.optimized_stages import (
        radar_channels_already_sized,
    )

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    rgb_tensor = rgb_tensor.to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if radar_channels_already_sized(radar_tensor, MODEL_WIDTH, MODEL_HEIGHT):
        radar_packed = np.ascontiguousarray(radar_tensor)
    else:
        radar_packed = np.ascontiguousarray(np.stack([
            cv2.resize(channel, (MODEL_WIDTH, MODEL_HEIGHT),
                       interpolation=(cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR))
            for index, channel in enumerate(radar_tensor)], axis=0))
    radar = torch.from_numpy(radar_packed).unsqueeze(0)
    return torch.cat(((rgb_tensor - mean) / std, radar.to(device=device, dtype=torch.float32)),
                     dim=1)


# ---------------------------------------------------------------------------
# components 1 and 2: the builder and the tracker
# ---------------------------------------------------------------------------


class BundleParkedModule:
    """The historical ``parked`` module with ``build_radar_sample`` routed through
    the validated profiled builder and the CUDA rasterizer; every other attribute
    (object-GT builders, constants) forwards unchanged."""

    def __init__(self, module: Any, rasterizer: Any) -> None:
        self._bundle_module = module
        self._bundle_rasterizer = rasterizer
        self.calls = 0

    def build_radar_sample(self, **kwargs: Any) -> tuple:
        from rl_agent.splitfusion_supervisor_analysis_v1.profiled_sensor_stages import (
            build_radar_sample_profiled,
        )

        require("rasterizer_override" not in kwargs, "caller must not pass an override")
        tensor, points, summary = build_radar_sample_profiled(
            rasterizer_override=self._bundle_rasterizer, **kwargs)
        self.calls += 1
        return tensor, points, {k: v for k, v in summary.items()
                                if not k.startswith(PROFILE_KEY_PREFIX)}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bundle_module, name)


def optimized_tracker_like(tracker: Any) -> Any:
    from rl_agent.splitfusion_sensor_optimization_v1.optimized_stages import (
        SingleSortStationaryTrackAccumulator,
    )

    require(int(tracker._keys.size) == 0,
            "the stationary tracker must be replaced before it accumulates state")
    return SingleSortStationaryTrackAccumulator(
        stationary_velocity_mps=float(tracker.stationary_velocity_mps),
        parked_threshold_s=float(tracker.parked_threshold_s),
        association_grid_m=float(tracker.association_grid_m),
        max_stale_s=float(tracker.max_stale_s))


def warm_rasterizer(rasterizer: Any) -> dict[str, Any]:
    """One tiny synthetic call so the first live frame does not pay CUDA setup."""
    started = time.perf_counter_ns()
    out = rasterizer(width=MODEL_WIDTH, height=MODEL_HEIGHT,
                     u=np.array([10.0, 700.0], dtype=np.float64),
                     v=np.array([20.0, 400.0], dtype=np.float64),
                     depth_m=np.array([5.0, 50.0], dtype=np.float64),
                     velocity_mps=np.array([0.0, -3.0], dtype=np.float32),
                     stationary_age_s=np.array([1.0, 0.0], dtype=np.float32),
                     valid_mask=np.array([True, True]), max_range_m=120.0,
                     max_abs_velocity_mps=20.0, parked_threshold_s=5.0, point_radius_px=4)
    return {"warm_ms": (time.perf_counter_ns() - started) / 1e6,
            "shape": list(out.shape), "dtype": str(out.dtype)}


def innermost_preparation_timer(parked: Any, timer_type: type) -> Any:
    """Walk the profiler's proxy chain down to the Phase-6 ``_PreparationTimer``."""
    current = parked
    for _ in range(6):
        if isinstance(current, timer_type):
            return current
        current = getattr(current, "_profiled_target", None)
        if current is None:
            break
    raise RadarBundleError("Phase-6 _PreparationTimer not found under the collector's parked proxy")


def install_packing_short_circuit(live_base: Any) -> dict[str, Any]:
    """Component 3.  Must run *before* the profiler wraps ``_prepare_live_input``
    so the ``ue.input_7ch`` span still brackets the packing."""
    original = live_base._prepare_live_input
    require(getattr(original, "__name__", "") == "_prepare_live_input",
            "packing seam must be installed before any other wrapper")
    live_base._prepare_live_input = prepare_live_input_identity_resize
    return {"packing_replaced": original.__name__}


def install_builder_and_tracker(*, pinned: Any, phase6_ue: Any, evidence_dir: Path,
                                record: Mapping[str, Any]) -> type:
    """Components 1 and 2 as an outer collector subclass (installed after the
    Phase-6 and profiler collectors).  The bundle record is written create-only
    once the live collector exists, i.e. before any frame."""
    from rl_agent.splitfusion_sensor_optimization_v1 import optimized_stages as OS

    base = pinned.PassiveSplitCollector

    class RadarBundleCollectorV1(base):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            rasterizer = OS.CudaRadarRasterizer(self.live.device)
            warm = warm_rasterizer(rasterizer)
            self.tracker = optimized_tracker_like(self.tracker)
            timer = innermost_preparation_timer(self.parked, phase6_ue._PreparationTimer)
            module = timer._parked
            require(not isinstance(module, BundleParkedModule), "bundle installed twice")
            timer._parked = BundleParkedModule(module, rasterizer)
            self._radar_bundle = {**record, "bundle": OPTIMIZED, "components": list(COMPONENTS),
                                  "optimized_stages": OS.__file__,
                                  "device": str(self.live.device), "rasterizer_warm": warm,
                                  "tracker": type(self.tracker).__name__,
                                  "tracker_params": {
                                      "stationary_velocity_mps": self.tracker.stationary_velocity_mps,
                                      "parked_threshold_s": self.tracker.parked_threshold_s,
                                      "association_grid_m": self.tracker.association_grid_m,
                                      "max_stale_s": self.tracker.max_stale_s},
                                  "installed_wall_ns": time.time_ns()}
            write_snapshot(evidence_dir, self._radar_bundle)

    pinned.PassiveSplitCollector = RadarBundleCollectorV1
    return RadarBundleCollectorV1


def record_production(evidence_dir: Path) -> dict[str, Any]:
    document = {"bundle": PRODUCTION, "components": [], "packing_replaced": None}
    write_snapshot(evidence_dir, document)
    return document


def write_snapshot(directory: Path, document: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SNAPSHOT_NAME
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(document), handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    return path

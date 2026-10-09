#!/usr/bin/env python3
"""Timestamp-binned logical radar sweeps and the 10 Hz prepared-input cadence.

Contract implemented here (Route B perception dataset v2):

* CARLA world / control tick: 20 Hz (``fixed_delta_seconds = 0.05``).
* Radar ``points_per_second`` stays at the measured 200,000. The sensor is left
  free-running (``sensor_tick = 0``) so it emits one raw callback per world
  tick; the logical sweep is rebuilt here from timestamps instead of being
  commanded through ``sensor_tick``.
* A logical sweep is a non-overlapping 100 ms half-open bin
  ``(anchor + 0.1*(i-1), anchor + 0.1*i]``. ``anchor`` is the timestamp of the
  first prepared-input tick, so a bin always closes exactly on the tick where a
  model input is due. Every raw callback whose timestamp falls in the bin is
  accumulated - the count of callbacks per bin is measured, never assumed.
* A prepared model input is built at 10 Hz from the current and the immediately
  previous logical sweep, i.e. contiguous 200 ms of support.
* Dataset persistence keeps every second prepared input -> 5 Hz.

Radar returns from the previous sweep were captured from a different ego pose.
They are lifted to world coordinates with *their own* callback transform at
ingest time and re-expressed in the current radar frame when the window is
requested, so the fused window is motion compensated. The window is returned in
the same ``[altitude, azimuth, depth, velocity]`` layout CARLA produces, which
means the existing ``build_radar_sample`` and every radar tensor shape/channel
are used unchanged.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_FUSION_PKG = Path(__file__).resolve().parent.parent / "pole_lraspp_multimodal_fusion"
if str(_FUSION_PKG) not in sys.path:
    sys.path.insert(0, str(_FUSION_PKG))

from pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    radar_raw_to_alt_az_depth_velocity,
    radar_spherical_to_world,
    transform_points,
)

WORLD_TICK_HZ = 20.0
WORLD_DELTA_S = 1.0 / WORLD_TICK_HZ
SWEEP_PERIOD_S = 0.1
WINDOW_SWEEPS = 2
PREPARE_EVERY_N_TICKS = 2          # 20 Hz world -> 10 Hz prepared input
SAVE_EVERY_N_PREPARED = 2          # 10 Hz prepared -> 5 Hz persisted
# Bin-boundary tolerance. CARLA reports elapsed time as a float32-derived double,
# so a nominal 0.05 s tick is 0.05000000074505806 s and the error against the sweep
# anchor accumulates without bound over an episode (~7e-7 s after 900 ticks). A
# 1e-6 s tolerance was measured to flip the boundary by a whole world tick partway
# through a 45 s run, splitting sweeps into one callback each. The tolerance is
# therefore a quarter of a world tick: far larger than any timestamp drift, and far
# smaller than the half-tick spacing between a boundary callback and an interior one.
BIN_EPSILON_S = 0.25 * WORLD_DELTA_S
RADAR_POINTS_PER_SECOND = 200000


class RadarSweepError(RuntimeError):
    """A logical-sweep or cadence invariant failed."""


@dataclass
class _Callback:
    frame: int
    timestamp_s: float
    returns: int
    world_velocity: np.ndarray  # (N, 4) world xyz + radial velocity, float32
    # As measured by CARLA in this callback's own sensor frame, before any
    # motion compensation: [altitude, azimuth, depth, radial velocity].
    detections: np.ndarray


@dataclass
class _Sweep:
    index: int
    callbacks: list[_Callback] = field(default_factory=list)

    @property
    def returns(self) -> int:
        return int(sum(cb.returns for cb in self.callbacks))

    @property
    def first_timestamp_s(self) -> float:
        return min(cb.timestamp_s for cb in self.callbacks)

    @property
    def last_timestamp_s(self) -> float:
        return max(cb.timestamp_s for cb in self.callbacks)


class RadarSweepAggregator:
    """Accumulate raw radar callbacks into timestamp-defined 100 ms sweeps."""

    def __init__(
        self,
        *,
        sweep_period_s: float = SWEEP_PERIOD_S,
        window_sweeps: int = WINDOW_SWEEPS,
        keep_sweeps: int = 4,
        bin_epsilon_s: float = BIN_EPSILON_S,
    ) -> None:
        if window_sweeps < 1:
            raise ValueError("window_sweeps must be >= 1")
        if keep_sweeps < window_sweeps + 1:
            raise ValueError("keep_sweeps must exceed window_sweeps")
        self.sweep_period_s = float(sweep_period_s)
        self.window_sweeps = int(window_sweeps)
        self.keep_sweeps = int(keep_sweeps)
        self.bin_epsilon_s = float(bin_epsilon_s)

        self._anchor_s: float | None = None
        self._pending: list[_Callback] = []
        self._sweeps: "OrderedDict[int, _Sweep]" = OrderedDict()
        self._seen_frames: set[int] = set()
        self._last_frame: int | None = None
        self._last_timestamp_s: float | None = None

        # Measured, never assumed.
        self.raw_callbacks = 0
        self.total_returns = 0
        self.duplicate_callbacks = 0
        self.out_of_order_callbacks = 0
        self.dropped_callback_frames = 0
        self.timestamp_reversals = 0
        self.returns_per_callback: list[int] = []
        self.callback_intervals_s: list[float] = []
        self.retired_sweeps: list[dict[str, Any]] = []

    # -- binning ---------------------------------------------------------
    @property
    def anchor_s(self) -> float | None:
        return self._anchor_s

    def set_anchor(self, anchor_timestamp_s: float) -> None:
        """Anchor bin boundaries so that ``anchor_timestamp_s`` closes a sweep."""
        if self._anchor_s is not None:
            return
        self._anchor_s = float(anchor_timestamp_s)
        pending, self._pending = self._pending, []
        for callback in pending:
            self._file(callback)

    def sweep_index_for(self, timestamp_s: float) -> int:
        if self._anchor_s is None:
            raise RadarSweepError("sweep anchor is not set yet")
        offset = (float(timestamp_s) - self._anchor_s) / self.sweep_period_s
        return int(math.ceil(offset - self.bin_epsilon_s))

    @property
    def expected_callbacks_per_sweep(self) -> int:
        return int(round(self.sweep_period_s / WORLD_DELTA_S))

    @property
    def expected_window_callbacks(self) -> int:
        return self.expected_callbacks_per_sweep * self.window_sweeps

    def sweep_window_bounds_s(self, index: int) -> tuple[float, float]:
        if self._anchor_s is None:
            raise RadarSweepError("sweep anchor is not set yet")
        end = self._anchor_s + self.sweep_period_s * float(index)
        return end - self.sweep_period_s, end

    # -- ingest ----------------------------------------------------------
    def ingest(self, measurement: Any) -> None:
        frame = int(getattr(measurement, "frame", -1))
        timestamp_s = float(getattr(measurement, "timestamp", float("nan")))
        if frame in self._seen_frames:
            self.duplicate_callbacks += 1
            return
        if self._last_frame is not None:
            if frame < self._last_frame:
                self.out_of_order_callbacks += 1
            elif frame > self._last_frame + 1:
                self.dropped_callback_frames += frame - self._last_frame - 1
        if self._last_timestamp_s is not None:
            if timestamp_s < self._last_timestamp_s:
                self.timestamp_reversals += 1
            else:
                self.callback_intervals_s.append(timestamp_s - self._last_timestamp_s)
        self._seen_frames.add(frame)
        self._last_frame = frame
        self._last_timestamp_s = timestamp_s

        detections = radar_raw_to_alt_az_depth_velocity(bytes(measurement.raw_data))
        sensor_matrix = np.array(measurement.transform.get_matrix(), dtype=np.float64)
        world_velocity = radar_spherical_to_world(detections, sensor_matrix).astype(np.float32, copy=False)

        callback = _Callback(
            frame=frame,
            timestamp_s=timestamp_s,
            returns=int(detections.shape[0]),
            world_velocity=world_velocity,
            detections=detections.astype(np.float32, copy=False),
        )
        self.raw_callbacks += 1
        self.total_returns += callback.returns
        self.returns_per_callback.append(callback.returns)
        if self._anchor_s is None:
            self._pending.append(callback)
        else:
            self._file(callback)

    def _file(self, callback: _Callback) -> None:
        index = self.sweep_index_for(callback.timestamp_s)
        sweep = self._sweeps.get(index)
        if sweep is None:
            sweep = _Sweep(index=index)
            self._sweeps[index] = sweep
        sweep.callbacks.append(callback)
        self._prune()

    def _prune(self) -> None:
        while len(self._sweeps) > self.keep_sweeps:
            _, sweep = self._sweeps.popitem(last=False)
            self.retired_sweeps.append(self.sweep_report(sweep.index, sweep))

    # -- reporting -------------------------------------------------------
    def sweep_report(self, index: int, sweep: _Sweep | None = None) -> dict[str, Any]:
        sweep = sweep if sweep is not None else self._sweeps.get(index)
        if sweep is None:
            return {"sweep_index": index, "present": False}
        start, end = self.sweep_window_bounds_s(index)
        return {
            "sweep_index": index,
            "present": True,
            "bin_start_s": start,
            "bin_end_s": end,
            "callbacks": len(sweep.callbacks),
            "returns": sweep.returns,
            "callback_frames": [cb.frame for cb in sweep.callbacks],
            "first_timestamp_s": sweep.first_timestamp_s,
            "last_timestamp_s": sweep.last_timestamp_s,
        }

    def all_sweep_reports(self) -> list[dict[str, Any]]:
        live = [self.sweep_report(index) for index in sorted(self._sweeps)]
        return sorted(self.retired_sweeps + live, key=lambda row: row["sweep_index"])

    def has_window(self, current_index: int) -> bool:
        return all(
            (current_index - offset) in self._sweeps
            for offset in range(self.window_sweeps)
        )

    # -- window construction --------------------------------------------
    def window_detections(
        self,
        current_index: int,
        *,
        sensor_inverse_matrix: np.ndarray,
        reference_timestamp_s: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the fused window in CARLA ``[alt, az, depth, velocity]`` layout.

        Points are re-expressed in the radar frame given by
        ``sensor_inverse_matrix`` (the pose at the prepared-input tick), so the
        contribution of the previous sweep is motion compensated.
        """
        indices = [current_index - offset for offset in range(self.window_sweeps - 1, -1, -1)]
        missing = [index for index in indices if index not in self._sweeps]
        if missing:
            raise RadarSweepError(
                f"incomplete temporal window at sweep {current_index}: missing sweeps {missing}"
            )
        sweeps = [self._sweeps[index] for index in indices]
        blocks = [cb.world_velocity for sweep in sweeps for cb in sweep.callbacks if cb.returns]
        if blocks:
            world_velocity = np.concatenate(blocks, axis=0).astype(np.float64, copy=False)
        else:
            world_velocity = np.zeros((0, 4), dtype=np.float64)

        local = transform_points(world_velocity[:, :3], np.asarray(sensor_inverse_matrix, dtype=np.float64))
        if local.size:
            depth = np.linalg.norm(local, axis=1)
            safe = np.maximum(depth, 1e-9)
            altitude = np.arcsin(np.clip(local[:, 2] / safe, -1.0, 1.0))
            azimuth = np.arctan2(local[:, 1], local[:, 0])
            detections = np.stack(
                [altitude, azimuth, depth, world_velocity[:, 3]], axis=1
            ).astype(np.float32, copy=False)
        else:
            detections = np.zeros((0, 4), dtype=np.float32)

        # Raw, as-measured provenance for each retained return, in the same row
        # order as ``detections``. Nothing here is derived from ground truth or
        # actor identity - it is only what the sensor reported and when.
        reference_s = (
            float(reference_timestamp_s)
            if reference_timestamp_s is not None
            else max((cb.timestamp_s for sweep in sweeps for cb in sweep.callbacks), default=0.0)
        )
        raw_blocks: list[np.ndarray] = []
        age_blocks: list[np.ndarray] = []
        offset_blocks: list[np.ndarray] = []
        for sweep in sweeps:
            # 0 = current logical sweep, 1 = immediately previous.
            offset = current_index - sweep.index
            for callback in sweep.callbacks:
                if not callback.returns:
                    continue
                raw_blocks.append(callback.detections)
                age_blocks.append(
                    np.full((callback.returns,), reference_s - callback.timestamp_s, dtype=np.float32)
                )
                offset_blocks.append(np.full((callback.returns,), offset, dtype=np.uint8))
        if raw_blocks:
            raw = np.concatenate(raw_blocks, axis=0)
            provenance = {
                "original_altitude_rad": raw[:, 0].astype(np.float32, copy=False),
                "original_azimuth_rad": raw[:, 1].astype(np.float32, copy=False),
                "original_range_m": raw[:, 2].astype(np.float32, copy=False),
                "radial_velocity_mps": raw[:, 3].astype(np.float32, copy=False),
                "observation_age_s": np.concatenate(age_blocks, axis=0),
                "sweep_offset": np.concatenate(offset_blocks, axis=0),
            }
        else:
            provenance = {
                name: np.zeros((0,), dtype=np.float32)
                for name in ("original_altitude_rad", "original_azimuth_rad",
                             "original_range_m", "radial_velocity_mps", "observation_age_s")
            }
            provenance["sweep_offset"] = np.zeros((0,), dtype=np.uint8)
        if provenance["original_range_m"].shape[0] != detections.shape[0]:
            raise RadarSweepError(
                "raw provenance rows "
                f"({provenance['original_range_m'].shape[0]}) do not align with window "
                f"detections ({detections.shape[0]})"
            )

        timestamps = [cb.timestamp_s for sweep in sweeps for cb in sweep.callbacks]
        frames = [cb.frame for sweep in sweeps for cb in sweep.callbacks]
        contiguous = indices == list(range(indices[0], indices[-1] + 1))
        meta = {
            "sweep_indices": indices,
            "sweeps_contiguous": bool(contiguous),
            "callback_frames": frames,
            "callbacks": len(frames),
            "returns_per_sweep": [sweep.returns for sweep in sweeps],
            "returns": int(detections.shape[0]),
            "window_first_timestamp_s": min(timestamps) if timestamps else None,
            "window_last_timestamp_s": max(timestamps) if timestamps else None,
            "window_span_s": (max(timestamps) - min(timestamps)) if timestamps else None,
            "window_support_s": self.sweep_period_s * self.window_sweeps,
            "reference_timestamp_s": reference_s,
            "raw_provenance": provenance,
        }
        return detections, meta


class CadencePlan:
    """Tick -> prepared-input -> saved-frame bookkeeping for the 20/10/5 Hz contract."""

    def __init__(
        self,
        *,
        prepare_every_n_ticks: int = PREPARE_EVERY_N_TICKS,
        save_every_n_prepared: int = SAVE_EVERY_N_PREPARED,
    ) -> None:
        self.prepare_every_n_ticks = int(prepare_every_n_ticks)
        self.save_every_n_prepared = int(save_every_n_prepared)
        self.camera_frame_parity: int | None = None
        self.ticks = 0
        self.prepared = 0
        self.saved = 0
        self.warmup_prepared_skipped = 0
        self.tick_timestamps_s: list[float] = []
        self.prepared_timestamps_s: list[float] = []
        self.saved_timestamps_s: list[float] = []

    def lock_camera_parity(self, camera_frame: int) -> None:
        self.camera_frame_parity = int(camera_frame) % self.prepare_every_n_ticks

    def is_prepare_frame(self, world_frame: int) -> bool:
        if self.camera_frame_parity is None:
            return False
        return int(world_frame) % self.prepare_every_n_ticks == self.camera_frame_parity

    def note_tick(self, timestamp_s: float) -> None:
        self.ticks += 1
        self.tick_timestamps_s.append(float(timestamp_s))

    def note_prepared(self, timestamp_s: float) -> bool:
        """Register a prepared input; return True when it must also be persisted."""
        self.prepared_timestamps_s.append(float(timestamp_s))
        must_save = (self.prepared % self.save_every_n_prepared) == 0
        self.prepared += 1
        if must_save:
            self.saved_timestamps_s.append(float(timestamp_s))
            self.saved += 1
        return must_save


def cadence_stats(timestamps_s: list[float]) -> dict[str, Any]:
    """Interval statistics in simulated seconds for a cadence stream."""
    if len(timestamps_s) < 2:
        return {
            "count": len(timestamps_s),
            "hz": None,
            "interval_s_min": None,
            "interval_s_mean": None,
            "interval_s_max": None,
            "reversals": 0,
        }
    intervals = [timestamps_s[i] - timestamps_s[i - 1] for i in range(1, len(timestamps_s))]
    mean = sum(intervals) / len(intervals)
    return {
        "count": len(timestamps_s),
        "hz": (1.0 / mean) if mean > 0 else None,
        "interval_s_min": min(intervals),
        "interval_s_mean": mean,
        "interval_s_max": max(intervals),
        "reversals": sum(1 for value in intervals if value < 0.0),
    }


def summarize(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": sum(numeric) / len(numeric),
        "max": max(numeric),
    }

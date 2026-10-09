#!/usr/bin/env python3
"""Bit-exact replacements for the dominant sensor-preparation stages.

Every implementation here is required to reproduce the production output
*exactly* -- same shape, dtype, ordering and bit pattern.  None of them changes
camera/radar synchronization, radar-window membership, sweep count, timestamps,
stationary-track semantics, coordinate frames, intrinsics/extrinsics, the
seven-channel tensor contract, the action identity, the model or the map path.

Measured localization (idle host, 40,000-point window, 768x448, radius 4):

* ``P12`` rasterization 6.81 ms, of which ``5 x np.maximum.at`` is 3.52 ms and
  ``5 x cv2.dilate`` is 1.61 ms.
* ``P09`` stationary tracking 3.56 ms, of which ``np.unique(return_inverse)``
  plus ``np.argsort(inverse)`` is 2.95 ms -- two sorts of the same data.
* ``P21`` radar resize/pack, which resizes 768x448 channels to 768x448.

The three candidates below remove exactly those costs and nothing else.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    FastStationaryTrackAccumulator,
)


class SingleSortStationaryTrackAccumulator(FastStationaryTrackAccumulator):
    """``FastStationaryTrackAccumulator`` with one stable sort instead of two.

    The production implementation calls ``np.unique(packed, return_inverse=True)``
    and then ``np.argsort(inverse, kind="stable")``.  ``inverse`` is a
    rank-preserving relabelling of ``packed``, so sorting by ``inverse`` and
    sorting by ``packed`` produce the *identical* stable permutation, and the
    unique keys, counts and group identifiers are all recoverable from that one
    permutation.  The second sort is therefore redundant.

    Matching, reset behaviour, within-cell point order, the retained track table
    and the returned ages are unchanged.  ``np.minimum.at`` is likewise replaced
    by a reversed last-write-wins assignment: within a group the stable order
    lists moving returns in ascending position, so writing them in reverse
    leaves the smallest position in place.
    """

    def update(self, world_velocity_points: np.ndarray, frame_time_s: float) -> np.ndarray:
        points = np.asarray(world_velocity_points)
        if points.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if not (points.ndim == 2 and points.shape[1] >= 4):
            raise ValueError("stationary tracker input must be [N,>=4]")
        now = float(frame_time_s)
        scale = max(0.05, self.association_grid_m)
        packed = self._pack_keys(
            points[:, 0].astype(np.float64, copy=False) / scale,
            points[:, 1].astype(np.float64, copy=False) / scale,
        )

        # The single stable sort. Everything below is derived from it.
        order = np.argsort(packed, kind="stable")
        sorted_keys = packed[order]
        if sorted_keys.size == 1:
            starts = np.zeros(1, dtype=np.int64)
        else:
            starts = np.concatenate((
                np.zeros(1, dtype=np.int64),
                np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]).astype(np.int64) + 1,
            ))
        unique_keys = sorted_keys[starts]
        counts = np.diff(
            np.concatenate((starts, np.array([sorted_keys.size], dtype=np.int64)))
        )
        group_ids = np.repeat(np.arange(len(unique_keys), dtype=np.int64), counts)

        stationary = (
            np.abs(points[:, 3].astype(np.float64, copy=False))
            <= self.stationary_velocity_mps
        )
        ages = np.zeros((points.shape[0],), dtype=np.float32)

        previous_age = np.zeros(len(unique_keys), dtype=np.float64)
        previous_seen = np.full(len(unique_keys), now, dtype=np.float64)
        if self._keys.size:
            prior_positions = np.searchsorted(self._keys, unique_keys)
            in_bounds = prior_positions < len(self._keys)
            matched = np.zeros(len(unique_keys), dtype=bool)
            matched[in_bounds] = (
                self._keys[prior_positions[in_bounds]] == unique_keys[in_bounds]
            )
            previous_age[matched] = self._ages[prior_positions[matched]]
            previous_seen[matched] = self._last_seen[prior_positions[matched]]
        dt = np.maximum(0.0, now - previous_seen)
        first_indices = order[starts]
        first_age = np.where(
            stationary[first_indices],
            np.minimum(self.parked_threshold_s * 3.0, previous_age + dt),
            0.0,
        )

        positions_in_group = np.arange(len(order), dtype=np.int64) - starts[group_ids]
        first_moving = counts.astype(np.int64, copy=True)
        moving_positions = np.flatnonzero(~stationary[order])
        if moving_positions.size:
            reversed_moving = moving_positions[::-1]
            first_moving[group_ids[reversed_moving]] = positions_in_group[reversed_moving]
        prefix = positions_in_group < first_moving[group_ids]
        ages[order[prefix]] = first_age[group_ids[prefix]].astype(np.float32)
        final_age = np.where(first_moving == counts, first_age, 0.0)
        last_indices = order[starts + counts - 1]

        retain = np.zeros(len(self._keys), dtype=bool)
        if self._keys.size:
            current_positions = np.searchsorted(unique_keys, self._keys)
            current_bounds = current_positions < len(unique_keys)
            is_current = np.zeros(len(self._keys), dtype=bool)
            is_current[current_bounds] = (
                unique_keys[current_positions[current_bounds]]
                == self._keys[current_bounds]
            )
            stale_after = max(self.max_stale_s, self.association_grid_m)
            retain = (~is_current) & ((now - self._last_seen) <= stale_after)
        merged_keys = np.concatenate((self._keys[retain], unique_keys))
        merged_order = np.argsort(merged_keys, kind="stable")
        self._keys = merged_keys[merged_order]
        self._ages = np.concatenate((self._ages[retain], final_age))[merged_order]
        self._last_seen = np.concatenate(
            (self._last_seen[retain], np.full(len(unique_keys), now, dtype=np.float64))
        )[merged_order]
        self._x = np.concatenate(
            (self._x[retain], points[last_indices, 0].astype(np.float64, copy=False))
        )[merged_order]
        self._y = np.concatenate(
            (self._y[retain], points[last_indices, 1].astype(np.float64, copy=False))
        )[merged_order]
        return ages


class CudaRadarRasterizer:
    """CUDA scatter-max + max-pool equivalent of ``rasterize_radar_channels_fast``.

    The production rasterizer scatters five per-pixel images with
    ``np.maximum.at`` and then dilates each with a square ``cv2`` kernel.  Both
    halves are pure maximum reductions, and a maximum over float32 values is
    order-independent and therefore bit-exact under any evaluation order.

    ``max_pool2d`` with ``kernel=2r+1``, ``stride=1`` and no padding over the
    ``r``-padded canvas produces exactly the interior crop the production code
    takes after dilation, so the border-handling question never arises: every
    output pixel's window lies wholly inside the padded canvas.

    All point filtering, rounding, score arithmetic and the signed-velocity
    recombination are performed with the identical NumPy expressions, so the
    only work moved to the device is the two maximum reductions.
    """

    def __init__(self, device: Any) -> None:
        import torch

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaRadarRasterizer requires a CUDA device")
        self._torch = torch

    def __call__(
        self,
        *,
        width: int,
        height: int,
        u: np.ndarray,
        v: np.ndarray,
        depth_m: np.ndarray,
        velocity_mps: np.ndarray,
        stationary_age_s: np.ndarray,
        valid_mask: np.ndarray,
        max_range_m: float,
        max_abs_velocity_mps: float,
        parked_threshold_s: float,
        point_radius_px: int = 2,
    ) -> np.ndarray:
        torch = self._torch
        channels = np.zeros((4, int(height), int(width)), dtype=np.float32)
        if u.size == 0:
            return channels
        in_image = (
            valid_mask
            & (u >= 0.0)
            & (u < float(width))
            & (v >= 0.0)
            & (v < float(height))
            & np.isfinite(depth_m)
        )
        selected = np.flatnonzero(in_image)
        if selected.size == 0:
            return channels

        px = np.rint(u[selected]).astype(np.int32, copy=False)
        py = np.rint(v[selected]).astype(np.int32, copy=False)
        max_range = max(1.0, float(max_range_m))
        max_velocity = max(0.1, float(max_abs_velocity_mps))
        parked_threshold = max(0.1, float(parked_threshold_s))
        range_score = (
            1.0
            - np.clip(depth_m[selected].astype(np.float32, copy=False), 0.0, max_range)
            / max_range
        ).astype(np.float32, copy=False)
        vel_score = np.clip(
            velocity_mps[selected].astype(np.float32, copy=False) / max_velocity,
            -1.0,
            1.0,
        ).astype(np.float32, copy=False)
        age_score = (
            np.clip(
                stationary_age_s[selected].astype(np.float32, copy=False),
                0.0,
                parked_threshold,
            )
            / parked_threshold
        ).astype(np.float32, copy=False)

        radius = max(0, int(point_radius_px))
        pad = radius
        scatter_h = int(height) + 2 * pad
        scatter_w = int(width) + 2 * pad
        scatter_y = py + pad
        scatter_x = px + pad
        # A rounded centre may land on ``width``/``height`` exactly; with no
        # padding that falls outside the canvas, exactly as the production
        # ``center_valid`` filter handles it.
        center_valid = (
            (scatter_x >= 0)
            & (scatter_x < scatter_w)
            & (scatter_y >= 0)
            & (scatter_y < scatter_h)
        )
        if not center_valid.all():
            keep = np.flatnonzero(center_valid)
            if keep.size == 0:
                return channels
            scatter_x = scatter_x[keep]
            scatter_y = scatter_y[keep]
            range_score = range_score[keep]
            vel_score = vel_score[keep]
            age_score = age_score[keep]

        flat = scatter_y.astype(np.int64) * scatter_w + scatter_x.astype(np.int64)
        source = np.stack(
            [
                np.ones_like(range_score),
                range_score,
                np.maximum(vel_score, 0.0),
                np.maximum(-vel_score, 0.0),
                age_score,
            ],
            axis=0,
        )
        index = torch.from_numpy(flat).to(self.device, non_blocking=True)
        values = torch.from_numpy(source).to(self.device, non_blocking=True)
        canvas = torch.zeros(
            (5, scatter_h * scatter_w), dtype=torch.float32, device=self.device
        )
        canvas.scatter_reduce_(
            1, index.unsqueeze(0).expand(5, -1), values, reduce="amax", include_self=True
        )
        view = canvas.view(1, 5, scatter_h, scatter_w)
        if radius > 0:
            pooled = torch.nn.functional.max_pool2d(
                view, kernel_size=2 * radius + 1, stride=1, padding=0
            )[0]
        else:
            pooled = view[0]
        occupancy = (pooled[0] > 0).to(torch.float32)
        positive, negative = pooled[2], pooled[3]
        velocity = torch.where(positive >= negative, positive, -negative)
        stacked = torch.stack([occupancy, pooled[1], velocity, pooled[4]], dim=0)
        return stacked.cpu().numpy()


def radar_channels_already_sized(radar_tensor: np.ndarray, width: int, height: int) -> bool:
    """True when the radar resize in seven-channel packing is the identity.

    ``build_radar_sample`` already rasterizes at the model size, so the resize
    in the packing step maps 768x448 onto 768x448.  ``cv2.resize`` to an
    identical size is bit-exact for both ``INTER_NEAREST`` and ``INTER_LINEAR``
    (the sampling grid coincides with the source grid), so skipping it changes
    no value.
    """

    return (
        radar_tensor.ndim == 3
        and int(radar_tensor.shape[1]) == int(height)
        and int(radar_tensor.shape[2]) == int(width)
    )

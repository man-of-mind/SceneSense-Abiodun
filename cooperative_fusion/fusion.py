#!/usr/bin/env python3
"""Cooperative position-fusion module (core contribution).

Each view contributes, per detected object:
  - camera center C (world, 3D)
  - a unit bearing ray d (world) toward the object (from the heatmap-peak pixel + intrinsics
    + camera orientation) -- PRECISE
  - a single-view world-position estimate p = C + depth*d -- depth is NOISY (~1.2 m, the
    measured single-view error), so p inherits that depth variance
  - confidence/score

Estimators (increasing principle):
  1. mean         : average the per-view world positions
  2. covariance   : confidence/precision-weighted average (information filter)
  3. triangulate  : least-squares closest point of the per-view bearing rays  <-- target

The thesis: bearings are precise, depth is noisy, so triangulating two bearings recovers a
much better position than averaging two depth-based estimates. This module + the Monte-Carlo
check below validate that before wiring to live two-view perception.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class ViewDetection:
    cam_center: np.ndarray      # (3,) world
    bearing: np.ndarray         # (3,) unit world ray to object
    world_pos: np.ndarray       # (3,) single-view estimate C + depth*d
    score: float = 1.0
    depth_std_m: float = 1.2    # assumed per-view depth uncertainty


def fuse_mean(dets: Sequence[ViewDetection]) -> np.ndarray:
    return np.mean([d.world_pos for d in dets], axis=0)


def fuse_covariance(dets: Sequence[ViewDetection]) -> np.ndarray:
    # Weight each view by 1/depth_var * score (scalar information weight).
    w = np.array([d.score / max(1e-6, d.depth_std_m ** 2) for d in dets])
    w = w / w.sum()
    return np.sum([wi * d.world_pos for wi, d in zip(w, dets)], axis=0)


def fuse_triangulate(dets: Sequence[ViewDetection]) -> np.ndarray:
    """Least-squares closest point to all bearing lines {C_i + t d_i}.
    Solve (sum (I - d d^T)) p = sum (I - d d^T) C_i."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for d in dets:
        di = d.bearing / max(1e-9, np.linalg.norm(d.bearing))
        P = np.eye(3) - np.outer(di, di)        # projects onto plane perp to the ray
        w = float(d.score)
        A += w * P
        b += w * P @ d.cam_center
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


# ----------------------------- synthetic validation -----------------------------
def _monte_carlo(n=20000, depth_std=1.2, bearing_std_deg=0.3, baseline_m=10.0, obj_range_m=15.0, seed=0):
    """Two static cameras separated by `baseline_m`, both viewing an object at ~obj_range_m.
    Bearings precise (small angular noise); depth noisy (depth_std). Compare estimators."""
    rng = np.random.default_rng(seed)
    errs = {"viewA": [], "viewB": [], "mean": [], "covariance": [], "triangulate": []}
    for _ in range(n):
        # object somewhere in front, two cameras offset laterally (different viewpoints)
        P = np.array([obj_range_m + rng.normal(0, 3), rng.normal(0, 4), 0.0])
        cams = [np.array([0.0, -baseline_m / 2, 1.5]), np.array([0.0, +baseline_m / 2, 1.5])]
        dets = []
        for C in cams:
            true = P - C
            true_range = np.linalg.norm(true)
            d_true = true / true_range
            # precise bearing + small angular noise
            ang = np.radians(rng.normal(0, bearing_std_deg, size=3))
            d = d_true + ang  # small perturbation
            d = d / np.linalg.norm(d)
            # single-view world estimate: precise bearing, NOISY depth
            depth_est = true_range + rng.normal(0, depth_std)
            p = C + depth_est * d
            dets.append(ViewDetection(C, d, p, score=1.0, depth_std_m=depth_std))
        errs["viewA"].append(np.linalg.norm(dets[0].world_pos[:2] - P[:2]))
        errs["viewB"].append(np.linalg.norm(dets[1].world_pos[:2] - P[:2]))
        errs["mean"].append(np.linalg.norm(fuse_mean(dets)[:2] - P[:2]))
        errs["covariance"].append(np.linalg.norm(fuse_covariance(dets)[:2] - P[:2]))
        errs["triangulate"].append(np.linalg.norm(fuse_triangulate(dets)[:2] - P[:2]))
    return {k: (float(np.mean(v)), float(np.median(v))) for k, v in errs.items()}


if __name__ == "__main__":
    print("Monte-Carlo cooperative fusion (XY position error vs GT), depth_std=1.2 m, bearing_std=0.3 deg\n")
    for base in (6.0, 12.0, 20.0):
        r = _monte_carlo(baseline_m=base)
        print(f"baseline {base:4.0f} m between views:")
        for k in ("viewA", "viewB", "mean", "covariance", "triangulate"):
            print(f"   {k:12s} MAE={r[k][0]:.3f} m  median={r[k][1]:.3f} m")
        print()
    print("Expect: single-view ~1.2 m (depth-limited); mean ~0.85 m (1/sqrt2); "
          "triangulate << that and improving with baseline (bearing-limited).")

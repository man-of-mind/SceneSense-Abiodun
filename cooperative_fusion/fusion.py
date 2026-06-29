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

import math
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
    dims: np.ndarray = None     # (3,) predicted (length, width, height) in OBJECT frame
    yaw: float = 0.0            # object world yaw (rad)


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


def fuse_dimensions(dets: Sequence[ViewDetection]) -> np.ndarray:
    """Fuse per-view 3D box dimensions (length, width, height) using viewing geometry.

    A view observes the extent PERPENDICULAR to its line of sight well, and the extent ALONG
    it poorly: a front-facing view (ray ~along the object's forward axis) measures WIDTH+HEIGHT
    but guesses LENGTH; a side view (ray ~along the lateral axis) measures LENGTH+HEIGHT but
    guesses WIDTH. So weight each view's length by how aligned its ray is with the lateral axis,
    and its width by alignment with the forward axis. Height is seen by all ground views.

    Returns fused (length, width, height). Recovers the full box from complementary views even
    though neither view alone sees all three extents.
    """
    yaws = np.array([d.yaw for d in dets], dtype=float)
    # circular mean of object yaw across views
    yaw = math.atan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
    fwd = np.array([math.cos(yaw), math.sin(yaw)])      # object forward (length) axis, ground
    lat = np.array([-math.sin(yaw), math.cos(yaw)])     # object lateral (width) axis, ground
    L = W = wL = wW = 0.0
    heights = []
    for d in dets:
        if d.dims is None:
            continue
        r = np.asarray(d.bearing, float)[:2]
        n = np.linalg.norm(r)
        if n < 1e-9:
            continue
        r = r / n
        a_len = abs(float(r @ lat))   # length well-observed when ray ~ along lateral (side view)
        a_wid = abs(float(r @ fwd))   # width well-observed when ray ~ along forward (front view)
        L += a_len * float(d.dims[0]); wL += a_len
        W += a_wid * float(d.dims[1]); wW += a_wid
        heights.append(float(d.dims[2]))
    length = L / wL if wL > 1e-6 else float(np.mean([d.dims[0] for d in dets if d.dims is not None]))
    width = W / wW if wW > 1e-6 else float(np.mean([d.dims[1] for d in dets if d.dims is not None]))
    height = float(np.mean(heights)) if heights else 0.0
    return np.array([length, width, height])


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

    # ---- dimension-fusion self-test ----
    print("\nDimension fusion (front view + side view -> full L x W x H):")
    rng = np.random.default_rng(1)
    true_L, true_W, true_H, yaw = 4.6, 1.9, 1.5, 0.0
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]); lat = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    n = 5000
    per_view_err = []; fused_err = []
    for _ in range(n):
        # View A ~front (ray ~ along forward): sees W,H well, L poorly.
        # View B ~side  (ray ~ along lateral): sees L,H well, W poorly.
        def noisy(obs_good, obs_bad_dim):
            d = np.array([true_L, true_W, true_H], float)
            d = d + rng.normal(0, 0.08, 3)          # small noise on all
            d[obs_bad_dim] = [true_L, true_W, true_H][obs_bad_dim] + rng.normal(0, 1.2)  # big noise on unobserved
            return d
        a = ViewDetection(np.zeros(3), fwd, np.zeros(3), dims=noisy(True, 0), yaw=yaw)   # front: bad on L (idx0)
        b = ViewDetection(np.zeros(3), lat, np.zeros(3), dims=noisy(True, 1), yaw=yaw)   # side:  bad on W (idx1)
        fused = fuse_dimensions([a, b])
        truth = np.array([true_L, true_W, true_H])
        per_view_err.append(np.mean(np.abs(a.dims - truth)))
        fused_err.append(np.mean(np.abs(fused - truth)))
    print(f"   single-view dim MAE = {np.mean(per_view_err):.3f} m   fused dim MAE = {np.mean(fused_err):.3f} m")
    print("   Expect: fused << single-view (each axis taken from the view that observes it).")

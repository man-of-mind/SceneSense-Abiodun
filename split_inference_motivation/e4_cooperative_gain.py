"""E4 - Cooperative gain: localization error + map coverage, single ego vs two egos.

Reframed per PLAN.md (2026-07-22): localization + coverage, NOT detection recall.

Two parts:
  P1 COVERAGE  - on REAL CARLA scene geometry (real GT object world positions, real
                 dimensions, real ego-A camera pose/FOV), how many objects can one ego
                 localize vs two? Ego B's pose is SYNTHESIZED at a controlled offset.
  P2 LOCALIZATION - for objects both egos see, triangulation vs single-view monocular,
                 using the validated fuse_triangulate() geometry from cooperative_fusion/.

Emits results/E4_raw.json + results/E4_coverage.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common_setup import AB, EXP_DIR

sys.path.insert(0, str(AB / "cooperative_fusion"))
from fusion import ViewDetection, fuse_mean, fuse_triangulate  # noqa: E402

OUT = Path(__file__).parent / "results"


# ----------------------------- geometry helpers -----------------------------
def yaw_forward(yaw_deg):
    y = math.radians(yaw_deg)
    return np.array([math.cos(y), math.sin(y)])


def in_fov(cam_xy, cam_yaw_deg, target_xy, fov_deg, max_range):
    v = target_xy - cam_xy
    r = float(np.linalg.norm(v))
    if r < 1e-6 or r > max_range:
        return False, r
    f = yaw_forward(cam_yaw_deg)
    cosang = float(np.dot(v / r, f))
    cosang = max(-1.0, min(1.0, cosang))
    return (math.degrees(math.acos(cosang)) <= fov_deg / 2.0), r


def is_occluded(cam_xy, target_xy, target_r, occluders):
    """2-D line-of-sight test: an occluder disc blocks the ray if it straddles the
    segment cam->target, lies strictly nearer than the target, and is not the target
    itself. Discs use radius = 0.5*max(size_x,size_y) from the real GT extents."""
    v = target_xy - cam_xy
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return False
    u = v / L
    for (oxy, orad) in occluders:
        d = oxy - cam_xy
        t = float(np.dot(d, u))
        if t <= 0.5 or t >= L - max(0.5, target_r):   # behind camera, or at/past the target
            continue
        perp = float(np.linalg.norm(d - t * u))
        if perp < orad:                                # ray passes through the occluder disc
            return True
    return False


def bearing_ray(cam_xyz, target_xyz, bearing_std_deg, rng):
    d = np.asarray(target_xyz, float) - np.asarray(cam_xyz, float)
    d = d / np.linalg.norm(d)
    d = d + np.radians(rng.normal(0, bearing_std_deg, size=3))
    return d / np.linalg.norm(d)


# ----------------------------- data loading -----------------------------
def load_scene_frames(limit_frames, min_objects, max_range):
    """Group real GT objects by sample_id and attach the real ego camera pose."""
    ds = EXP_DIR / "dataset"
    poses = {}
    with open(ds / "manifest.csv") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "test":
                continue
            poses[row["sample_id"]] = {
                "xy": np.array([float(row["camera_x"]), float(row["camera_y"])]),
                "z": float(row["camera_z"]),
                "yaw": float(row["camera_yaw"]),
                "fov": float(row["camera_fov"]),
            }
    by_sample = defaultdict(list)
    with open(ds / "object_boxes.csv") as fh:
        for row in csv.DictReader(fh):
            sid = row["sample_id"]
            if sid not in poses:
                continue
            try:
                by_sample[sid].append({
                    "label": row["label"],
                    "xy": np.array([float(row["object_world_x"]), float(row["object_world_y"])]),
                    "z": float(row["object_world_z"]),
                    "radius": 0.5 * max(float(row["gt_size_x_m"]), float(row["gt_size_y_m"])),
                    "dist": float(row["gt_distance_m"]),
                })
            except (ValueError, KeyError):
                continue
    frames = []
    for sid, objs in by_sample.items():
        near = [o for o in objs if o["dist"] <= max_range]
        if len(near) >= min_objects:
            frames.append((sid, poses[sid], objs))
        if len(frames) >= limit_frames:
            break
    return frames


# ----------------------------- P1: coverage -----------------------------
def coverage_experiment(frames, baseline_m, fov_deg, max_range, lateral_frac=0.7):
    """Ego B synthesized at a controlled offset from ego A's REAL pose:
    lateral_frac of the baseline sideways (adjacent/oncoming lane) and the rest
    forward, yawed to face ego A's scene centre."""
    tot = dict(A=0, B=0, union=0, both=0, none=0, objects=0)
    per_frame = []
    for sid, pose, objs in frames:
        camA = pose["xy"]
        yawA = pose["yaw"]
        f = yaw_forward(yawA)
        lat = np.array([-f[1], f[0]])
        camB = camA + lat * (baseline_m * lateral_frac) + f * (baseline_m * (1 - lateral_frac))
        # ego B looks at the centroid of the near objects (a peer covering the same area)
        near = [o for o in objs if o["dist"] <= max_range]
        if not near:
            continue
        centre = np.mean([o["xy"] for o in near], axis=0)
        vB = centre - camB
        yawB = math.degrees(math.atan2(vB[1], vB[0]))

        discs = [(o["xy"], o["radius"]) for o in objs]
        cA = cB = cU = cBoth = 0
        for o in near:
            occl = [(xy, r) for (xy, r) in discs if not np.allclose(xy, o["xy"])]
            okA, _ = in_fov(camA, yawA, o["xy"], fov_deg, max_range)
            if okA:
                okA = not is_occluded(camA, o["xy"], o["radius"], occl)
            okB, _ = in_fov(camB, yawB, o["xy"], fov_deg, max_range)
            if okB:
                okB = not is_occluded(camB, o["xy"], o["radius"], occl)
            cA += okA
            cB += okB
            cU += (okA or okB)
            cBoth += (okA and okB)
        n = len(near)
        tot["objects"] += n
        tot["A"] += cA
        tot["B"] += cB
        tot["union"] += cU
        tot["both"] += cBoth
        tot["none"] += n - cU
        per_frame.append(dict(sample_id=sid, n=n, A=cA, B=cB, union=cU, both=cBoth))
    return tot, per_frame


# ----------------------------- P2: localization -----------------------------
def localization_experiment(frames, baseline_m, fov_deg, max_range,
                            depth_std, bearing_std_deg, seed=0, lateral_frac=0.7):
    """For objects visible to BOTH egos, compare single-view (precise bearing + noisy
    depth) against two-view triangulation, on real object/ego geometry."""
    rng = np.random.default_rng(seed)
    errs = defaultdict(list)
    for sid, pose, objs in frames:
        camA_xy, yawA = pose["xy"], pose["yaw"]
        f = yaw_forward(yawA)
        lat = np.array([-f[1], f[0]])
        camB_xy = camA_xy + lat * (baseline_m * lateral_frac) + f * (baseline_m * (1 - lateral_frac))
        near = [o for o in objs if o["dist"] <= max_range]
        if not near:
            continue
        centre = np.mean([o["xy"] for o in near], axis=0)
        vB = centre - camB_xy
        yawB = math.degrees(math.atan2(vB[1], vB[0]))
        discs = [(o["xy"], o["radius"]) for o in objs]
        camA = np.array([camA_xy[0], camA_xy[1], pose["z"]])
        camB = np.array([camB_xy[0], camB_xy[1], pose["z"]])
        for o in near:
            occl = [(xy, r) for (xy, r) in discs if not np.allclose(xy, o["xy"])]
            okA = in_fov(camA_xy, yawA, o["xy"], fov_deg, max_range)[0] and \
                not is_occluded(camA_xy, o["xy"], o["radius"], occl)
            okB = in_fov(camB_xy, yawB, o["xy"], fov_deg, max_range)[0] and \
                not is_occluded(camB_xy, o["xy"], o["radius"], occl)
            if not (okA and okB):
                continue
            P = np.array([o["xy"][0], o["xy"][1], o["z"]])
            dets = []
            for C in (camA, camB):
                d = bearing_ray(C, P, bearing_std_deg, rng)
                true_range = float(np.linalg.norm(P - C))
                p = C + (true_range + rng.normal(0, depth_std)) * d
                dets.append(ViewDetection(C, d, p, score=1.0, depth_std_m=depth_std))
            errs["single_A"].append(np.linalg.norm(dets[0].world_pos[:2] - P[:2]))
            errs["single_B"].append(np.linalg.norm(dets[1].world_pos[:2] - P[:2]))
            errs["mean_2view"].append(np.linalg.norm(fuse_mean(dets)[:2] - P[:2]))
            errs["triangulate"].append(np.linalg.norm(fuse_triangulate(dets)[:2] - P[:2]))
    return {k: dict(mae=float(np.mean(v)), median=float(np.median(v)), n=len(v))
            for k, v in errs.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--min-objects", type=int, default=3)
    ap.add_argument("--max-range", type=float, default=40.0, help="detection gate (m)")
    ap.add_argument("--fov", type=float, default=120.0)
    ap.add_argument("--depth-std", type=float, default=1.2)
    ap.add_argument("--bearing-std-deg", type=float, default=0.3)
    ap.add_argument("--baselines", type=float, nargs="+", default=[4, 8, 14, 20])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print(f"loading real GT scenes (test split, gate {args.max_range} m, >={args.min_objects} objects)...")
    frames = load_scene_frames(args.frames, args.min_objects, args.max_range)
    print(f"  {len(frames)} frames, "
          f"{sum(len([o for o in objs if o['dist']<=args.max_range]) for _,_,objs in frames)} in-range objects")

    results = {"config": vars(args), "n_frames": len(frames), "coverage": {}, "localization": {}}

    print(f"\n== P1 COVERAGE (real GT layouts; ego B synthesized at offset) ==")
    print(f"{'baseline':>9} {'objects':>8} {'ego A':>12} {'ego B':>12} {'UNION(coop)':>13} {'BOTH(triang)':>13}")
    for b in args.baselines:
        tot, per_frame = coverage_experiment(frames, b, args.fov, args.max_range)
        n = max(1, tot["objects"])
        row = {k: tot[k] for k in tot}
        row.update({f"{k}_pct": round(100 * tot[k] / n, 2) for k in ("A", "B", "union", "both", "none")})
        results["coverage"][str(b)] = row
        print(f"{b:9.0f} {tot['objects']:8d} {tot['A']:7d} ({row['A_pct']:5.1f}%) "
              f"{tot['B']:7d} ({row['B_pct']:5.1f}%) "
              f"{tot['union']:7d} ({row['union_pct']:5.1f}%) "
              f"{tot['both']:7d} ({row['both_pct']:5.1f}%)")

    print(f"\n== P2 LOCALIZATION (objects visible to BOTH; depth_std={args.depth_std} m, "
          f"bearing_std={args.bearing_std_deg} deg) ==")
    print(f"{'baseline':>9} {'n':>6} {'singleA':>9} {'singleB':>9} {'mean2v':>9} {'TRIANG':>9} {'gain':>7}")
    for b in args.baselines:
        loc = localization_experiment(frames, b, args.fov, args.max_range,
                                      args.depth_std, args.bearing_std_deg)
        if not loc:
            print(f"{b:9.0f}  (no objects visible to both)")
            continue
        results["localization"][str(b)] = loc
        best_single = min(loc["single_A"]["mae"], loc["single_B"]["mae"])
        gain = best_single / loc["triangulate"]["mae"]
        print(f"{b:9.0f} {loc['triangulate']['n']:6d} {loc['single_A']['mae']:9.3f} "
              f"{loc['single_B']['mae']:9.3f} {loc['mean_2view']['mae']:9.3f} "
              f"{loc['triangulate']['mae']:9.3f} {gain:6.2f}x")

    (OUT / "E4_raw.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUT/'E4_raw.json'}")
    return results


if __name__ == "__main__":
    main()

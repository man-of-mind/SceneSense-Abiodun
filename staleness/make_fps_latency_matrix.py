#!/usr/bin/env python3
"""FPS × Latency combined visualization: grouped bar chart showing error at each (FPS, latency) pair."""
import csv, glob, math, sys, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

FPS_SWEEP = [1, 5, 10, 15, 20, 25, 30]
LATENCIES = [0.0, 0.050, 0.105, 0.267]
LATENCY_LABELS = ["Y=0 (ideal)", "loopback 50ms", "AE-128 105ms", "no-AE 267ms"]
LATENCY_COLORS = ["#2ca02c", "#009E73", "#0072B2", "#D55E00"]
BANDS = [(0, 4, "~walk/slow"), (4, 8, "~6 mph"), (8, 12, "~10 mph"), (12, 16, "~14 mph"),
         (16, 20, "~18 mph"), (20, 26, "~23 mph"), (26, 30, "~28 mph"), (30, 40, "~32 mph")]
MIN_N = 15
OUT = Path("staleness/plots"); OUT.mkdir(parents=True, exist_ok=True)

def truthy(v): return str(v).strip().lower() in ("true", "1", "yes")

def gt_at(sm, t):
    if t <= sm[0][0]: return sm[0][1], sm[0][2]
    if t >= sm[-1][0]:
        (t0, x0, y0), (t1, x1, y1) = sm[-2][:3], sm[-1][:3]; dt = t1 - t0
        return (x1 + (x1 - x0) * ((t - t1) / dt), y1 + (y1 - y0) * ((t - t1) / dt)) if dt > 1e-6 else (sm[-1][1], sm[-1][2])
    for i in range(1, len(sm)):
        if sm[i][0] >= t:
            (t0, x0, y0), (t1, x1, y1) = sm[i - 1][:3], sm[i][:3]; k = (t - t0) / max(1e-6, t1 - t0)
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return sm[-1][1], sm[-1][2]

def _is_sweep(run):
    for m in glob.glob(run + "/streams/*metrics.csv"):
        try:
            rows = list(csv.DictReader(open(m)))
            if rows and str(rows[0].get("run_group", "")).startswith("speedsweep_"):
                return True
        except: pass
    return False

runs = [r for r in sorted(glob.glob("staleness/metrics_logs/scenesense_runs/*")) if _is_sweep(r)]
print(f"speed-sweep runs found: {len(runs)}")

all_obs = []
for run in runs:
    gt = list(csv.DictReader(open(glob.glob(run + "/streams/*ground_truth.csv")[0])))
    pr = list(csv.DictReader(open(glob.glob(run + "/streams/*predictions.csv")[0])))
    traj = defaultdict(list)
    for r in gt:
        try:
            traj[r["actor_id"]].append((float(r["carla_timestamp"]), float(r.get("origin_x") or r["world_x"]),
                                        float(r.get("origin_y") or r["world_y"]), int(r["frame_id"]),
                                        truthy(r.get("in_camera_frustum", "")), float(r.get("distance_m", 999))))
        except: pass
    for a in traj: traj[a].sort()
    prby = defaultdict(list)
    for r in pr:
        try:
            if float(r.get("score", 0)) >= 0.2: prby[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
        except: pass
    for aid, sm in traj.items():
        for i, (t, x, y, fid, inf, dist) in enumerate(sm):
            if not (inf and dist <= 25.0): continue
            P = prby.get(fid, [])
            if not P: continue
            d = min(P, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            if math.hypot(d[0] - x, d[1] - y) > 2.0: continue
            j = min(max(1, i), len(sm) - 1); (t0, x0, y0), (t1, x1, y1) = sm[j - 1][:3], sm[j][:3]
            v = math.hypot(x1 - x0, y1 - y0) / max(1e-6, t1 - t0) * 2.237
            all_obs.append((v, t, sm, d))

print(f"observations: {len(all_obs)}")

representative_bands = [(8, 12, "~10 mph"), (16, 20, "~18 mph"), (30, 40, "~32 mph")]

for lo, hi, speed_label in representative_bands:
    sel = [(v, t, sm, d) for v, t, sm, d in all_obs if lo <= v < hi]
    if len(sel) < MIN_N:
        print(f"Skipping {speed_label} (only {len(sel)} observations)")
        continue
    
    print(f"\n=== {speed_label} ===")
    
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    
    x_pos = np.arange(len(FPS_SWEEP))
    bar_width = 0.2
    
    for lat_idx, (lat_sec, lat_label, lat_color) in enumerate(zip(LATENCIES, LATENCY_LABELS, LATENCY_COLORS)):
        errors = []
        for fps_val in FPS_SWEEP:
            total_lag = lat_sec + 1.0/fps_val
            errs = [math.hypot(d[0] - gt_at(sm, t + total_lag)[0], d[1] - gt_at(sm, t + total_lag)[1]) 
                    for v, t, sm, d in sel]
            errors.append(statistics.mean(errs))
            print(f"  FPS={fps_val:2d}, {lat_label:15s}: {errors[-1]:.2f} m")
        
        offset = (lat_idx - 1.5) * bar_width
        ax.bar(x_pos + offset, errors, bar_width, label=lat_label, color=lat_color, alpha=0.85, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Update Rate (FPS)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Localization Error (m)", fontsize=12, fontweight="bold")
    ax.set_title(f"FPS × Latency: {speed_label}", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(f) for f in FPS_SWEEP])
    ax.set_ylim(0, None)
    ax.legend(fontsize=10, frameon=True, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    fname = f"fps_latency_matrix_{speed_label.replace(' ', '').replace('~', '')}"
    fig.savefig(OUT/f"{fname}.pdf", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT}/{fname}.pdf")

print("\nDone.")

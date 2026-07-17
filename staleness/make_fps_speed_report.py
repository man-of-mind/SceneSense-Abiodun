#!/usr/bin/env python3
"""FPS as spatial-map staleness: localization error vs target speed, per FPS value."""
import csv, glob, math, sys, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

FPS_SWEEP = [1, 5, 10, 15, 20, 25, 30]
FPS_COLORS = {1: "#e41a1c", 5: "#377eb8", 10: "#4daf4a", 15: "#984ea3", 20: "#ff7f00", 25: "#a65628", 30: "#f781bf"}
OPS = [("loopback ~50ms", 0.050), ("AE-128 ~105ms", 0.105), ("no-AE ~267ms", 0.267)]
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

print(f"\n{'band':12s} {'#':>4s} " + " ".join(f"  {fps:>2d}fps  " for fps in FPS_SWEEP))
fps_curves = {}
for bi, (lo, hi, lab) in enumerate(BANDS):
    sel = [(v, t, sm, d) for v, t, sm, d in all_obs if lo <= v < hi]
    if len(sel) < MIN_N:
        print(f"{lab:12s} {len(sel):>4d}  (skipped)")
        continue
    print(f"{lab:12s} {len(sel):>4d} ", end="")
    for fps_idx, fps_val in enumerate(FPS_SWEEP):
        errs = [math.hypot(d[0] - gt_at(sm, t + 1.0/fps_val)[0], d[1] - gt_at(sm, t + 1.0/fps_val)[1]) for v, t, sm, d in sel]
        err_mean = statistics.mean(errs)
        if fps_val not in fps_curves: fps_curves[fps_val] = []
        fps_curves[fps_val].append((0.5*(lo+hi), err_mean))
        print(f"{err_mean:>6.2f}", end=" ")
    print()

fig, ax = plt.subplots(figsize=(8.2, 5.6))
for fps_val in FPS_SWEEP:
    if fps_val in fps_curves:
        speeds, errs = zip(*fps_curves[fps_val])
        ax.plot(speeds, errs, color=FPS_COLORS[fps_val], lw=2.4, marker="o", ms=5, label=f"{fps_val} FPS")

ax.set_xlim(0, 35); ax.set_ylim(0, None); ax.margins(x=0)
ax.set_xlabel("target object speed (mph)", fontsize=11); ax.set_ylabel("localization error (m)", fontsize=11)
ax.set_title("Map staleness at ideal latency (Y=0)", fontweight="bold", fontsize=12)
ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT/"fps_mapStaleness_worstcase.pdf", bbox_inches="tight", dpi=200)
fig.savefig(OUT/"fps_mapStaleness_worstcase.png", dpi=200, bbox_inches="tight")
print(f"\nwrote {OUT}/fps_mapStaleness_worstcase.pdf")

print("\n=== FPS plateau analysis ===")
print(f"{'Speed band':12s} {'5→10 FPS':>10s} {'10→20 FPS':>10s} {'20→30 FPS':>10s} {'Saturates?':>12s}")
for lo, hi, lab in BANDS:
    if any(f in fps_curves for f in [5, 10, 20, 30]):
        sel = [(v, t, sm, d) for v, t, sm, d in all_obs if lo <= v < hi]
        if len(sel) >= MIN_N:
            err_5 = statistics.mean([math.hypot(d[0] - gt_at(sm, t + 1.0/5)[0], d[1] - gt_at(sm, t + 1.0/5)[1]) for v, t, sm, d in sel])
            err_10 = statistics.mean([math.hypot(d[0] - gt_at(sm, t + 1.0/10)[0], d[1] - gt_at(sm, t + 1.0/10)[1]) for v, t, sm, d in sel])
            err_20 = statistics.mean([math.hypot(d[0] - gt_at(sm, t + 1.0/20)[0], d[1] - gt_at(sm, t + 1.0/20)[1]) for v, t, sm, d in sel])
            err_30 = statistics.mean([math.hypot(d[0] - gt_at(sm, t + 1.0/30)[0], d[1] - gt_at(sm, t + 1.0/30)[1]) for v, t, sm, d in sel])
            imp_5_10 = err_5 - err_10
            imp_10_20 = err_10 - err_20
            imp_20_30 = err_20 - err_30
            saturated = "yes (≤0.05m)" if imp_20_30 < 0.05 else "no"
            print(f"{lab:12s} {imp_5_10:>9.2f}m  {imp_10_20:>9.2f}m  {imp_20_30:>9.2f}m  {saturated:>12s}")

for op_label, op_latency_sec in OPS:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for fps_val in FPS_SWEEP:
        band_speeds = []
        band_errors = []
        total_lag = op_latency_sec + 1.0/fps_val
        for bi, (lo, hi, lab) in enumerate(BANDS):
            sel = [(v, t, sm, d) for v, t, sm, d in all_obs if lo <= v < hi]
            if len(sel) >= MIN_N:
                errs = [math.hypot(d[0] - gt_at(sm, t + total_lag)[0], d[1] - gt_at(sm, t + total_lag)[1]) for v, t, sm, d in sel]
                band_speeds.append(0.5*(lo+hi))
                band_errors.append(statistics.mean(errs))
        if band_errors:
            ax.plot(band_speeds, band_errors, color=FPS_COLORS[fps_val], lw=2.4, marker="o", ms=5, label=f"{fps_val} FPS")

    ax.set_xlim(0, 35); ax.set_ylim(0, None); ax.margins(x=0)
    ax.set_xlabel("target object speed (mph)", fontsize=11); ax.set_ylabel("localization error (m)", fontsize=11)
    fname_safe = op_label.lower().replace(" ", "_").replace("~", "").replace("ms", "")
    ax.set_title(f"FPS + {op_label}", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/f"fps_latency_{fname_safe}.pdf", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT}/fps_latency_{fname_safe}.pdf")

print(f"\nKey: FPS is a latency analog. Static cars flat; fast cars scale as v/FPS. Gains saturate at 20-25 FPS.")

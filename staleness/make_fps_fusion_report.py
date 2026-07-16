#!/usr/bin/env python3
"""FPS & TEMPORAL ACCUMULATION vs latency, per target speed. Single-frame (memoryless) vs temporal-accumulation
(recursive constant-velocity Kalman: at each frame it folds in ALL prior frames of the object's track -- recent
frames weighted more via process noise -- and predicts FORWARD by Y to cancel staleness). Higher FPS packs the
accumulation into a fresher window -> better velocity -> better prediction. Pools moving-ego FPS runs (egofps_*
mid + egofpsfast_* ~30mph), tags by capture FPS, bins by MEASURED instantaneous speed. GT = ORIGIN convention;
near-range (<=25m) + 2 m gate = clean floor. Reports accumulation DEPTH (track lengths). Usage: make_fps_fusion_report.py
"""
import csv, glob, math, re, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

GATE = 2.0; NEAR = 25.0; SCORE = 0.2; MIN_N = 15
Y_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.269]
# speed bins (m/s, label). center-labelled in mph.
BINS = [(0.5, 2.5, "pedestrian"), (5.0, 8.0, "~14 mph"), (8.0, 11.0, "~22 mph"), (11.0, 18.0, "~30 mph")]
# distinct, colourblind-safe hues per FPS (Okabe-Ito) -- NOT shades of one colour
FPS_COL = {5: "#E69F00", 10: "#009E73", 15: "#56B4E9", 20: "#0072B2", 30: "#CC79A7"}
OUT = Path("staleness/plots"); OUT.mkdir(parents=True, exist_ok=True)

def truthy(v): return str(v).strip().lower() in ("true", "1", "yes")

def gt_at(sm, t):
    if t <= sm[0][0]: return sm[0][1], sm[0][2]
    if t >= sm[-1][0]:
        (t0, x0, y0), (t1, x1, y1) = sm[-2][:3], sm[-1][:3]; dt = t1 - t0
        return (x1 + (x1-x0)*((t-t1)/dt), y1 + (y1-y0)*((t-t1)/dt)) if dt > 1e-6 else (sm[-1][1], sm[-1][2])
    for i in range(1, len(sm)):
        if sm[i][0] >= t:
            (t0, x0, y0), (t1, x1, y1) = sm[i-1][:3], sm[i][:3]; k = (t - t0) / max(1e-6, t1 - t0)
            return x0 + (x1-x0)*k, y0 + (y1-y0)*k
    return sm[-1][1], sm[-1][2]

def band(v_mph):
    for lo, hi, lab in BINS:
        if lo*2.237 <= v_mph < hi*2.237: return lab
    return None

# pool egofps_<FPS> + egofpsfast_<FPS> per capture FPS
runs = defaultdict(list)
for d in sorted(glob.glob("staleness/metrics_logs/scenesense_runs/2026*")):
    mc = glob.glob(d + "/streams/*metrics.csv")
    if not mc: continue
    rows = list(csv.DictReader(open(mc[0])))
    if not rows: continue
    m = re.match(r"egofps(?:fast)?_(\d+)$", str(rows[0].get("run_group", "")))
    if m: runs[int(m.group(1))].append(d)
print("FPS runs pooled:", {f: len(v) for f, v in sorted(runs.items())})

R = np.diag([1.2**2, 1.2**2]); qa = 3.0**2
H = np.array([[1,0,0,0],[0,1,0,0]], float)
# agg[lab][fps][Y] = {"sf":[], "tr":[]}; depth[lab][fps] = [track lengths]
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"sf": [], "tr": []})))
depth = defaultdict(lambda: defaultdict(list))

for fps, dirs in runs.items():
    for run in dirs:
        gt = list(csv.DictReader(open(glob.glob(run + "/streams/*ground_truth.csv")[0])))
        pr = list(csv.DictReader(open(glob.glob(run + "/streams/*predictions.csv")[0])))
        traj = defaultdict(list)
        for r in gt:
            try: traj[r["actor_id"]].append((float(r["carla_timestamp"]), float(r.get("origin_x") or r["world_x"]),
                                             float(r.get("origin_y") or r["world_y"]), int(r["frame_id"]),
                                             truthy(r.get("in_camera_frustum", "")), float(r.get("distance_m", 999))))
            except: pass
        for a in traj: traj[a].sort()
        prby = defaultdict(list)
        for r in pr:
            try:
                if float(r.get("score", 0)) >= SCORE: prby[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
            except: pass
        for aid, sm in traj.items():
            dets = []
            for i, (t, x, y, fid, inf, dist) in enumerate(sm):
                if not (inf and dist <= NEAR): continue
                P = prby.get(fid, [])
                if not P: continue
                z = min(P, key=lambda p: math.hypot(p[0]-x, p[1]-y))
                if math.hypot(z[0]-x, z[1]-y) > GATE: continue
                j = min(max(1, i), len(sm)-1); (t0,x0,y0),(t1,x1,y1) = sm[j-1][:3], sm[j][:3]
                v = math.hypot(x1-x0, y1-y0) / max(1e-6, t1-t0)
                dets.append((t, z[0], z[1], v, sm))
            if len(dets) < 2: continue
            lab0 = band(statistics.median(d[3] for d in dets) * 2.237)
            if lab0: depth[lab0][fps].append(len(dets))    # accumulation depth = track length
            x = np.array([dets[0][1], dets[0][2], 0, 0], float); P = np.diag([R[0,0], R[1,1], 25, 25]); tprev = dets[0][0]
            for (t, zx, zy, v, sm) in dets:
                dt = t - tprev; tprev = t
                if dt > 0:
                    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
                    G = np.array([dt*dt/2, dt*dt/2, dt, dt]); Q = np.outer(G, G) * qa
                    x = F @ x; P = F @ P @ F.T + Q
                z = np.array([zx, zy]); S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ (z - H @ x); P = (np.eye(4) - K @ H) @ P
                lab = band(v * 2.237)
                if lab is None: continue
                for Y in Y_SWEEP:
                    gx, gy = gt_at(sm, t + Y)
                    agg[lab][fps][Y]["sf"].append(math.hypot(zx - gx, zy - gy))
                    px, py = x[0] + x[2]*Y, x[1] + x[3]*Y
                    agg[lab][fps][Y]["tr"].append(math.hypot(px - gx, py - gy))

# which (bin,FPS) cells are usable
print("\naccumulation DEPTH (frames fused per track) + #observations, by speed x FPS:")
for lo, hi, lab in BINS:
    if lab not in agg: continue
    print(f"  {lab}:")
    for fps in sorted(agg[lab]):
        n = len(agg[lab][fps].get(0.0, {}).get("sf", [])); dl = depth[lab][fps]
        dd = f"depth med={int(statistics.median(dl))} max={max(dl)}" if dl else "depth -"
        print(f"    {fps:>2} FPS: obs={n:>4}  {dd}")

# per speed with >=2 FPS having enough data -> one 2-panel figure
plotted = []
for lo, hi, lab in BINS:
    fps_ok = [f for f in sorted(agg[lab]) if len(agg[lab][f].get(0.0, {}).get("sf", [])) >= MIN_N]
    if len(fps_ok) < 2:
        if lab in agg: print(f"  (skip plot '{lab}': only {len(fps_ok)} FPS with >= {MIN_N} obs)")
        continue
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for f in fps_ok:
        cell = agg[lab][f]; c = FPS_COL.get(f, "#555")
        tr = [statistics.mean(cell[Y]["tr"]) for Y in Y_SWEEP]
        sf = [statistics.mean(cell[Y]["sf"]) for Y in Y_SWEEP]
        ax1.plot([Y*1000 for Y in Y_SWEEP], tr, color=c, ls="-", lw=2.4, marker="o", ms=5, label=f"{f} FPS")
        ax1.plot([Y*1000 for Y in Y_SWEEP], sf, color=c, ls=":", lw=1.5, alpha=0.85)
    ax1.plot([], [], color="#555", ls="-", lw=2.4, label="temporal accumulation (solid)")
    ax1.plot([], [], color="#555", ls=":", lw=1.5, label="single-frame (dotted)")
    ax1.set_xlim(0, Y_SWEEP[-1]*1000); ax1.margins(x=0); ax1.set_ylim(0, None); ax1.grid(alpha=0.25)
    ax1.set_xlabel("latency Y (ms) = capture→inference"); ax1.set_ylabel("localization error (m)")
    ax1.set_title(f"{lab}: temporal accumulation vs single-frame, by FPS", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    # right: error vs FPS at 269ms, single-frame vs temporal accumulation
    YHI = 0.269
    sfv = [statistics.mean(agg[lab][f][YHI]["sf"]) for f in fps_ok]
    trv = [statistics.mean(agg[lab][f][YHI]["tr"]) for f in fps_ok]
    ax2.plot(fps_ok, sfv, color="#999", ls="--", lw=2.0, marker="s", ms=6, label="single-frame")
    ax2.plot(fps_ok, trv, color="#0072B2", ls="-", lw=2.6, marker="o", ms=7, label="temporal accumulation")
    for f, s, tv in zip(fps_ok, sfv, trv):
        if tv < s: ax2.annotate(f"−{s-tv:.2f} m", (f, tv), textcoords="offset points", xytext=(0, -15), fontsize=8.5, color="#0072B2", ha="center")
    ax2.set_xlim(min(fps_ok)-1, max(fps_ok)+1); ax2.set_ylim(0, None); ax2.grid(alpha=0.25)
    ax2.set_xlabel("camera FPS (real CARLA capture)"); ax2.set_ylabel(f"{lab} error @ 269 ms (m)")
    ax2.set_title(f"{lab} @ 269 ms: accumulation gain grows with FPS", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9.5, frameon=False)
    fig.suptitle(f"Temporal accumulation (Kalman): fuses all past frames + predicts forward to cancel staleness — {lab}", fontsize=11.5, y=1.02)
    fig.tight_layout()
    tag = lab.replace("~", "").replace(" ", "").replace("/", "")
    fig.savefig(OUT / f"fps_fusion_{tag}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fps_fusion_{tag}.png", dpi=200, bbox_inches="tight")
    plotted.append(lab)
    print(f"\n=== {lab} (single-frame / temporal-accumulation, m) ===")
    print(f"{'FPS':>4s} " + " ".join(f"{int(Y*1000):>9d}ms" for Y in Y_SWEEP))
    for f in fps_ok:
        cell = agg[lab][f]
        print(f"{f:>4d} " + " ".join(f"{statistics.mean(cell[Y]['sf']):>4.2f}/{statistics.mean(cell[Y]['tr']):<4.2f}" for Y in Y_SWEEP))
print(f"\nwrote plots for: {plotted}  -> staleness/plots/fps_fusion_<speed>.pdf/.png")

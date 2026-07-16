#!/usr/bin/env python3
"""Per-SPEED x per-FPS: localization error vs latency, single-frame vs Kalman-tracked, for a few fixed speeds.
Answers Abiodun's design: fix a target speed, vary camera FPS, show how error(Y) behaves -- the FPS benefit
appears in the TRACKED curves (single-frame is FPS-independent at a given latency; it's the flat baseline).
Pools all moving-ego FPS runs (egofps_* mid regime + egofpsfast_* ~30mph regime), tags each by capture FPS,
bins observations by MEASURED instantaneous speed. Recursive constant-velocity Kalman: fuses past frames +
predicts forward by Y to cancel staleness (higher FPS -> fresher velocity -> better prediction).
GT uses ORIGIN convention. Near-range (<=NEAR) + tight gate = clean floor. Usage: make_fps_speed_report.py
"""
import csv, glob, math, re, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

GATE = 2.5; NEAR = 25.0; SCORE = 0.2
Y_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.269]
BANDS = [(0, 4, "walk / slow (~0-4 mph)"), (11, 17, "~14 mph"), (26, 40, "~30 mph")]
MIN_N = 12
OPS = [105, 267]          # operating latencies to summarise the fusion gain at (AE-128, no-AE)
FPS_COL = {5: "#bdd7e7", 10: "#6baed6", 20: "#2171b5", 30: "#08306b"}
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

def band_of(mph):
    for lo, hi, lab in BANDS:
        if lo <= mph < hi: return lab
    return None

# find FPS runs (both regimes), tag by capture FPS
runs = {}  # fps -> list of dirs
for d in sorted(glob.glob("staleness/metrics_logs/scenesense_runs/2026*")):
    mc = glob.glob(d + "/streams/*metrics.csv")
    if not mc: continue
    rows = list(csv.DictReader(open(mc[0])))
    if not rows: continue
    m = re.match(r"egofps(?:fast)?_(\d+)$", str(rows[0].get("run_group", "")))
    if m: runs.setdefault(int(m.group(1)), []).append(d)
print("FPS runs:", {f: len(v) for f, v in sorted(runs.items())})

R = np.diag([1.2**2, 1.2**2]); qa = 3.0**2
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
# agg[band][fps][Y] = {"sf":[], "tr":[]}
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"sf": [], "tr": []})))

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
                z = min(P, key=lambda p: math.hypot(p[0] - x, p[1] - y))
                if math.hypot(z[0] - x, z[1] - y) > GATE: continue
                j = min(max(1, i), len(sm) - 1); (t0, x0, y0), (t1, x1, y1) = sm[j-1][:3], sm[j][:3]
                v = math.hypot(x1 - x0, y1 - y0) / max(1e-6, t1 - t0)
                dets.append((t, z[0], z[1], v, sm))
            if len(dets) < 2: continue
            x = np.array([dets[0][1], dets[0][2], 0, 0], float); P = np.diag([R[0,0], R[1,1], 25, 25]); tprev = dets[0][0]
            for (t, zx, zy, v, sm) in dets:
                dt = t - tprev; tprev = t
                if dt > 0:
                    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
                    G = np.array([dt*dt/2, dt*dt/2, dt, dt]); Q = np.outer(G, G) * qa
                    x = F @ x; P = F @ P @ F.T + Q
                z = np.array([zx, zy]); S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ (z - H @ x); P = (np.eye(4) - K @ H) @ P
                lab = band_of(v * 2.237)
                if lab is None: continue
                for Y in Y_SWEEP:
                    gx, gy = gt_at(sm, t + Y)
                    agg[lab][fps][Y]["sf"].append(math.hypot(zx - gx, zy - gy))
                    px, py = x[0] + x[2]*Y, x[1] + x[3]*Y
                    agg[lab][fps][Y]["tr"].append(math.hypot(px - gx, py - gy))

# ---- one panel per speed band: tracked (solid) + single-frame (dashed) vs Y, one colour per FPS ----
active = [(lo, hi, lab) for lo, hi, lab in BANDS if lab in agg]
fig, axes = plt.subplots(1, len(active), figsize=(5.4*len(active), 5.2), squeeze=False)
for ax, (lo, hi, lab) in zip(axes[0], active):
    for fps in sorted(agg[lab]):
        cell = agg[lab][fps]
        if len(cell.get(0.0, {}).get("tr", [])) < MIN_N: continue
        tr = [statistics.mean(cell[Y]["tr"]) for Y in Y_SWEEP]
        sf = [statistics.mean(cell[Y]["sf"]) for Y in Y_SWEEP]
        c = FPS_COL.get(fps, "#888")
        ax.plot([Y*1000 for Y in Y_SWEEP], tr, color=c, ls="-", lw=2.3, marker="o", ms=4, label=f"{fps} FPS (tracked)")
        ax.plot([Y*1000 for Y in Y_SWEEP], sf, color=c, ls="--", lw=1.3, alpha=0.7)
    ax.set_xlim(0, Y_SWEEP[-1]*1000); ax.margins(x=0); ax.set_ylim(0, None); ax.grid(alpha=0.25)
    ax.set_xlabel("latency Y (ms) = capture→inference"); ax.set_ylabel("localization error (m)")
    ax.set_title(lab, fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
axes[0][0].plot([], [], color="#555", ls="--", lw=1.3, label="single-frame (dashed)")
axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Per-speed: tracked error(Y) by FPS (solid) vs single-frame (dashed). FPS helps via fusion; slow=flat, fast=big gain",
             fontsize=11.5, y=1.02)
fig.tight_layout(); fig.savefig(OUT/"fps_speed_requirement.pdf", bbox_inches="tight")
fig.savefig(OUT/"fps_speed_requirement.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/fps_speed_requirement.pdf/.png")

# ---- tables: per band, per FPS, single-frame/tracked at each Y; + fusion gain at operating latencies ----
for lo, hi, lab in active:
    print(f"\n=== {lab} ===  (single-frame / tracked, m)")
    print(f"{'FPS':>4s} {'n':>4s} " + " ".join(f"{int(Y*1000):>9d}ms" for Y in Y_SWEEP))
    for fps in sorted(agg[lab]):
        cell = agg[lab][fps]; n = len(cell.get(0.0, {}).get("tr", []))
        if n < MIN_N: continue
        cells = " ".join(f"{statistics.mean(cell[Y]['sf']):>4.2f}/{statistics.mean(cell[Y]['tr']):<4.2f}" for Y in Y_SWEEP)
        print(f"{fps:>4d} {n:>4d} {cells}")
    print("  fusion gain (single-frame − tracked) at operating latency:")
    for opms in OPS:
        Yk = min(Y_SWEEP, key=lambda Y: abs(Y*1000 - opms))
        line = []
        for fps in sorted(agg[lab]):
            cell = agg[lab][fps]
            if len(cell.get(Yk, {}).get("tr", [])) < MIN_N: continue
            g = statistics.mean(cell[Yk]["sf"]) - statistics.mean(cell[Yk]["tr"])
            line.append(f"{fps}FPS:{g:+.2f}m")
        print(f"    ~{opms}ms: " + "  ".join(line))

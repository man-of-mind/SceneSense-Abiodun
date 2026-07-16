#!/usr/bin/env python3
"""Per-INDIVIDUAL-SPEED localization error vs latency Y, from OPPORTUNITY WINDOWS across the speed-sweep runs.
Method (agreed with Abiodun): a moving car-height ego drives among NPC traffic whose speed regime is varied per
run; ANY vehicle that enters good range (<=NEAR m, in camera frustum) and matches a prediction is an observation.
We bin each observation by its MEASURED instantaneous world speed (not a preset), then for each speed band compute
error(Y)=||pred(t) - GT_origin(t+Y)|| averaged over observations. GT uses the ORIGIN convention (matches training).
Latency Y = capture->inference (front+uplink+back); we overlay the measured operating points.
Usage: make_speed_error_report.py [NEAR_m=25] [gate_m=2.0] [score=0.2]   (auto-globs speedsweep_* runs)
"""
import csv, glob, math, sys, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

NEAR = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
GATE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
SCORE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
Y_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.269]
OPS = [("loopback ~50ms", 50, "#009E73"), ("AE-128 OAI ~105ms", 105, "#0072B2"), ("no-AE OAI ~267ms", 267, "#D55E00")]
# individual-speed bands (mph): centre-labelled, kept only if enough samples
# NOTE: Town10 traffic has an occupancy valley at 20-26 mph (cars cruise ~18 or jump to 26+), so the ~23 band
# is wider to gather enough samples; 22 mph itself is dynamically sparse (would need a constant-velocity target).
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
print(f"speed-sweep runs found: {len(runs)}  -> " + ", ".join(Path(r).name[-8:] for r in runs))

# collect opportunity-window observations: (speed_mph, [err(Y) for Y in Y_SWEEP])
obs = []
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
            if float(r.get("score", 0)) >= SCORE: prby[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
        except: pass
    for aid, sm in traj.items():
        for i, (t, x, y, fid, inf, dist) in enumerate(sm):
            if not (inf and dist <= NEAR): continue
            P = prby.get(fid, [])
            if not P: continue
            d = min(P, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            if math.hypot(d[0] - x, d[1] - y) > GATE: continue
            j = min(max(1, i), len(sm) - 1); (t0, x0, y0), (t1, x1, y1) = sm[j - 1][:3], sm[j][:3]
            v = math.hypot(x1 - x0, y1 - y0) / max(1e-6, t1 - t0)
            errs = [math.hypot(d[0] - gt_at(sm, t + Y)[0], d[1] - gt_at(sm, t + Y)[1]) for Y in Y_SWEEP]
            obs.append((v * 2.237, errs))
print(f"opportunity-window observations (<{NEAR:.0f}m, gate {GATE}m): {len(obs)}")

# aggregate by band
fig, ax = plt.subplots(figsize=(8.2, 5.6))
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(BANDS)))
print(f"\n{'band':12s} {'n':>4s} " + " ".join(f"{int(Y*1000):>4d}ms" for Y in Y_SWEEP))
rows = []
for bi, (lo, hi, lab) in enumerate(BANDS):
    sel = [errs for mph, errs in obs if lo <= mph < hi]
    if len(sel) < MIN_N:
        print(f"{lab:12s} {len(sel):>4d}  (skipped, <{MIN_N})")
        continue
    curve = [statistics.mean(e[k] for e in sel) for k in range(len(Y_SWEEP))]
    rows.append((lab, len(sel), curve))
    print(f"{lab:12s} {len(sel):>4d} " + " ".join(f"{c:>5.2f}" for c in curve) + " m")
    ax.plot([Y*1000 for Y in Y_SWEEP], curve, color=cmap[bi], lw=2.3, marker="o", ms=4, label=lab)

_ytop = ax.get_ylim()[1]
for lab, Yms, col in OPS:
    ax.axvline(Yms, color=col, ls=":", lw=1.6, alpha=0.85)
    ax.text(Yms+3, _ytop*0.60, lab, rotation=90, va="top", ha="left", fontsize=7.5, color=col)
ax.set_xlim(0, Y_SWEEP[-1]*1000); ax.margins(x=0); ax.set_ylim(0, None)
ax.set_xlabel("latency Y (ms) = capture→inference"); ax.set_ylabel("measured localization error (m)")
ax.set_title("Localization error vs latency, per target speed (opportunity windows)", fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="upper left"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT/"speed_error_requirement.pdf", bbox_inches="tight")
fig.savefig(OUT/"speed_error_requirement.png", dpi=200, bbox_inches="tight")
print(f"\nwrote {OUT}/speed_error_requirement.pdf/.png")

# ---- Requirement table: max latency Y (ms) to keep error <= eps, per speed. Interp the measured curve;
#      '>269' = still under eps at the largest measured Y; '—' = floor already exceeds eps (model-limited). ----
def max_Y_for(curve, eps):
    ys = [Y*1000 for Y in Y_SWEEP]
    if curve[0] > eps: return "—"          # floor already over budget: model-limited, not latency-limited
    if curve[-1] <= eps: return f">{ys[-1]:.0f}"
    for k in range(1, len(curve)):
        if curve[k] > eps:
            y0, y1, e0, e1 = ys[k-1], ys[k], curve[k-1], curve[k]
            return f"{y0 + (y1-y0)*(eps-e0)/(e1-e0):.0f}"
    return f">{ys[-1]:.0f}"
print("\nMax latency Y (ms) to keep localization error <= eps  (— = model floor already exceeds eps):")
EPS = [1.5, 2.0, 2.5, 3.0]
print(f"{'speed':12s} " + " ".join(f"e<={e}m" for e in EPS))
for lab, n, curve in rows:
    print(f"{lab:12s} " + " ".join(f"{max_Y_for(curve, e):>6s}" for e in EPS))

#!/usr/bin/env python3
"""Per-object staleness analysis (supervisor's method) from a natural-traffic run.

For every GT vehicle/pedestrian actor that is DETECTED (a prediction matches it near its GT), and for a sweep
of latency Y, compute the localization error the supervisor defined:
    error(Y) = || inferred_world_loc(captured at t) - GT_world_loc(t + Y) ||
where GT(t+Y) is the actor's own trajectory interpolated Y seconds later. Y=0 => model-error floor;
error grows ~ v*Y (staleness). Aggregate by object SPEED bin -> error-vs-Y curves + the latency threshold
that keeps error <= eps. FPS is handled separately (inter-frame gap = 1/FPS adds up to v/FPS on top of Y).

Usage: analyze_staleness.py <run_dir> [match_gate_m=4.0]
"""
import csv, sys, math
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1]); gate = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
gtf = run / "streams"
gt_csv = next(gtf.glob("*object_ground_truth.csv")); pr_csv = next(gtf.glob("*object_predictions.csv"))
gt = list(csv.DictReader(open(gt_csv))); pr = list(csv.DictReader(open(pr_csv)))

# per-actor GT trajectory: actor_id -> sorted [(t, x, y, class)]
traj = defaultdict(list)
for r in gt:
    try:
        traj[r["actor_id"]].append((float(r["carla_timestamp"]), float(r["world_x"]), float(r["world_y"]),
                                    r.get("class_name", "?"), int(r["frame_id"])))
    except Exception:
        pass
for a in traj:
    traj[a].sort()

# predictions per frame
prby = defaultdict(list)
for r in pr:
    try:
        prby[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
    except Exception:
        pass

def gt_at(samples, t):
    """Interpolate/extrapolate actor world pos at time t from its (t,x,y) samples."""
    if not samples:
        return None
    if t <= samples[0][0]:
        return samples[0][1], samples[0][2]
    if t >= samples[-1][0]:
        # extrapolate with last velocity
        if len(samples) >= 2:
            (t0, x0, y0), (t1, x1, y1) = samples[-2][:3], samples[-1][:3]
            dt = t1 - t0
            if dt > 1e-6:
                k = (t - t1) / dt
                return x1 + (x1 - x0) * k, y1 + (y1 - y0) * k
        return samples[-1][1], samples[-1][2]
    for i in range(1, len(samples)):
        if samples[i][0] >= t:
            (t0, x0, y0), (t1, x1, y1) = samples[i-1][:3], samples[i][:3]
            k = (t - t0) / max(1e-6, t1 - t0)
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return samples[-1][1], samples[-1][2]

def speed_at(samples, i):
    if i == 0 or i >= len(samples):
        i = min(max(1, i), len(samples) - 1)
    (t0, x0, y0), (t1, x1, y1) = samples[i-1][:3], samples[i][:3]
    dt = t1 - t0
    return math.hypot(x1 - x0, y1 - y0) / dt if dt > 1e-6 else 0.0

Y_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.269]   # s; incl. AE-128u4(0.105) region + no-AE(0.267)
SPEED_BINS = [(0.5, 2.5, "ped ~1-2 m/s"), (2.5, 6.0, "~5-13 mph"), (6.0, 10.0, "~13-22 mph"),
              (10.0, 15.0, "~22-33 mph"), (15.0, 22.0, "~33-49 mph"), (22.0, 99.0, "50+ mph")]
# accumulator: bin_idx -> Y -> list(errors)
acc = defaultdict(lambda: defaultdict(list))
acc_disp = defaultdict(lambda: defaultdict(list))   # PURE staleness = target displacement ||GT(t+Y)-GT(t)||
nmatch = 0
for aid, samples in traj.items():
    for i, (t, x, y, cls, fid) in enumerate(samples):
        preds = prby.get(fid, [])
        if not preds:
            continue
        d = min(preds, key=lambda p: math.hypot(p[0] - x, p[1] - y))
        if math.hypot(d[0] - x, d[1] - y) > gate:   # not detected this frame
            continue
        nmatch += 1
        v = speed_at(samples, i)
        b = next((bi for bi, (lo, hi, _) in enumerate(SPEED_BINS) if lo <= v < hi), None)
        if b is None:
            continue
        g0 = gt_at(samples, t)
        for Y in Y_SWEEP:
            g = gt_at(samples, t + Y)
            if g is None:
                continue
            acc[b][Y].append(math.hypot(d[0] - g[0], d[1] - g[1]))
            if g0 is not None:
                acc_disp[b][Y].append(math.hypot(g[0] - g0[0], g[1] - g0[1]))

print(f"run={run.name}  matched detections={nmatch}  actors={len(traj)}")
print(f"\n{'speed bin':<16} {'n':>5} | " + " ".join(f"Y={int(Y*1000):>3}ms" for Y in Y_SWEEP))
for bi, (lo, hi, label) in enumerate(SPEED_BINS):
    if bi not in acc:
        continue
    row = acc[bi]
    n0 = len(row.get(0.0, []))
    cells = []
    for Y in Y_SWEEP:
        vals = row.get(Y, [])
        cells.append(f"{sum(vals)/len(vals):>6.2f}" if vals else "   -  ")
    print(f"{label:<16} {n0:>5} | " + " ".join(cells) + "  m (mean loc err)")
print("\n--- PURE staleness = target displacement during Y (v*Y; the latency-budget driver, model-independent) ---")
print(f"{'speed bin':<16} {'n':>5} | " + " ".join(f"Y={int(Y*1000):>3}ms" for Y in Y_SWEEP))
for bi, (lo, hi, label) in enumerate(SPEED_BINS):
    if bi not in acc_disp:
        continue
    row = acc_disp[bi]
    n0 = len(row.get(Y_SWEEP[-1], []))
    cells = []
    for Y in Y_SWEEP:
        vals = row.get(Y, [])
        cells.append(f"{sum(vals)/len(vals):>6.2f}" if vals else "   -  ")
    print(f"{label:<16} {n0:>5} | " + " ".join(cells) + "  m displacement")
print("\nLatency budget: at speed v, staleness = v*Y. To keep staleness <= eps: Y <= eps/v.")
print("  eps=0.5m: ped(1.5)=333ms  city(8)=63ms  suburb(14)=36ms  hwy(30)=17ms")
print("  eps=1.0m: ped(1.5)=667ms  city(8)=125ms suburb(14)=71ms  hwy(30)=33ms")
print("(Total error = model floor + staleness combined vectorially; pure staleness is the clean threshold driver.)")
